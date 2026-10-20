//! Conservative, single-group KV selection over the existing placement index.
//!
//! This is a library building block, not a registered Router policy. Callers
//! provide engine-owned identities and fresh load observations, and establish
//! trust only after a validated snapshot, complete replay or clear barrier.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::{Locality, MatchQuery, RoutableTarget, StorageTier, TieredMatchProvider};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvRecoveryState {
    /// Subscribed or reattached, but no authoritative complete view exists yet.
    Recovering,
    /// A fresh snapshot, complete replay or clear barrier established the view.
    Healthy,
    /// The source is unavailable or stale without evidence of a sequence gap.
    Degraded,
    /// Ordering/identity evidence is inconsistent; live deltas cannot restore it.
    Unreconciled,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvSourceStatus {
    pub state: KvRecoveryState,
    pub reason: &'static str,
    pub incarnation: u64,
    pub sequence: Option<u64>,
}

#[derive(Debug, Clone)]
struct SourceState {
    incarnation: u64,
    sequence: Option<u64>,
    recovery: KvRecoveryState,
    observed_at: Instant,
    reason: &'static str,
}

/// Bounded, rank-scoped lifecycle state. One registry belongs to one cache
/// identity; sharing it across models/tokenizers/salts is not supported.
#[derive(Debug)]
pub struct KvTrustRegistry {
    sources: HashMap<RoutableTarget, SourceState>,
    capacity: usize,
}

impl KvTrustRegistry {
    pub fn new(capacity: usize) -> Result<Self, &'static str> {
        if capacity == 0 {
            return Err("trust registry capacity must be positive");
        }
        Ok(Self {
            sources: HashMap::new(),
            capacity,
        })
    }

    /// Attach/restart a source. Even reattaching the same incarnation revokes
    /// trust: receiving live deltas is not evidence of a complete baseline.
    /// The same incarnation keeps its observed sequence watermark so an older
    /// snapshot cannot roll the source back after reattachment.
    pub fn register(&mut self, target: RoutableTarget, incarnation: u64, now: Instant) -> bool {
        if !self.sources.contains_key(&target) && self.sources.len() >= self.capacity {
            return false;
        }
        if let Some(state) = self.sources.get_mut(&target) {
            if state.incarnation == incarnation {
                state.recovery = KvRecoveryState::Recovering;
                state.reason = "unreconciled";
                return true;
            }
        }
        self.sources.insert(
            target,
            SourceState {
                incarnation,
                sequence: None,
                recovery: KvRecoveryState::Recovering,
                observed_at: now,
                reason: "unreconciled",
            },
        );
        true
    }

    pub fn remove(&mut self, target: &RoutableTarget) {
        self.sources.remove(target);
    }

    /// Called only after the ingestion owner verifies a complete baseline.
    /// An obsolete incarnation or barrier behind observed deltas is rejected.
    /// This method must never be called from untrusted request metadata.
    pub fn reconcile(
        &mut self,
        target: &RoutableTarget,
        incarnation: u64,
        sequence: u64,
        now: Instant,
    ) -> bool {
        let Some(state) = self.sources.get_mut(target) else {
            return false;
        };
        if state.incarnation != incarnation {
            state.recovery = KvRecoveryState::Unreconciled;
            state.reason = "incarnation_mismatch";
            return false;
        }
        if state.sequence.is_some_and(|last| sequence < last) {
            state.recovery = KvRecoveryState::Unreconciled;
            state.reason = "reconciliation_rollback";
            return false;
        }
        state.sequence = Some(sequence);
        state.observed_at = now;
        state.recovery = KvRecoveryState::Healthy;
        state.reason = "verified";
        true
    }

    /// Observe an already validated/applied engine event. Duplicates do not
    /// refresh freshness; gaps, rollbacks, and epoch mismatches revoke trust.
    /// Replay deltas alone never establish or restore a baseline.
    pub fn advance(
        &mut self,
        target: &RoutableTarget,
        incarnation: u64,
        sequence: u64,
        now: Instant,
    ) -> bool {
        let Some(state) = self.sources.get_mut(target) else {
            return false;
        };
        if state.incarnation != incarnation {
            state.recovery = KvRecoveryState::Unreconciled;
            state.reason = "incarnation_mismatch";
            return false;
        }
        if state.sequence == Some(sequence) {
            return false;
        }
        if let Some(last) = state.sequence {
            if sequence < last {
                state.recovery = KvRecoveryState::Unreconciled;
                state.reason = "sequence_rollback";
                return false;
            }
            if last.checked_add(1) != Some(sequence) {
                state.recovery = KvRecoveryState::Unreconciled;
                state.reason = "sequence_gap_or_rollback";
            }
        }
        // A contiguous live delta may repair the local index, but it is not an
        // authoritative snapshot. A degraded or fenced source stays untrusted
        // until the ingestion owner supplies a fresh reconciliation barrier.
        if state.recovery != KvRecoveryState::Healthy
            && state.recovery != KvRecoveryState::Unreconciled
        {
            state.recovery = KvRecoveryState::Unreconciled;
            state.reason = "recovery_required";
        }
        state.sequence = Some(sequence);
        state.observed_at = now;
        true
    }

    pub fn mark_unavailable(&mut self, target: &RoutableTarget) {
        if let Some(state) = self.sources.get_mut(target) {
            // Availability is weaker evidence than ordering corruption. Keep
            // the original unreconciled cause visible until a barrier repairs it.
            if state.recovery == KvRecoveryState::Unreconciled {
                return;
            }
            state.recovery = KvRecoveryState::Degraded;
            state.reason = "unavailable";
        }
    }

    /// Live or replay evidence cannot establish exactness until a complete
    /// baseline has been reconstructed. This transition makes that ordering
    /// failure explicit and prevents freshness from masking it.
    pub fn mark_unreconciled(&mut self, target: &RoutableTarget, reason: &'static str) {
        if let Some(state) = self.sources.get_mut(target) {
            state.recovery = KvRecoveryState::Unreconciled;
            state.reason = reason;
        }
    }

    /// Project freshness without mutating the source. A stale source is
    /// degraded until an authoritative barrier restores a complete view.
    pub fn status(
        &self,
        target: &RoutableTarget,
        now: Instant,
        max_age: Duration,
    ) -> Option<KvSourceStatus> {
        let state = self.sources.get(target)?;
        let fresh = now
            .checked_duration_since(state.observed_at)
            .is_some_and(|age| age <= max_age);
        let (recovery, reason) = if state.recovery == KvRecoveryState::Healthy && !fresh {
            (KvRecoveryState::Degraded, "stale")
        } else {
            (state.recovery, state.reason)
        };
        Some(KvSourceStatus {
            state: recovery,
            reason,
            incarnation: state.incarnation,
            sequence: state.sequence,
        })
    }

    fn failure(
        &self,
        target: &RoutableTarget,
        now: Instant,
        max_age: Duration,
    ) -> Option<&'static str> {
        match self.status(target, now, max_age) {
            None => Some("unregistered"),
            Some(status) if status.state == KvRecoveryState::Healthy => None,
            Some(status) => Some(status.reason),
        }
    }
}

#[derive(Debug, Clone)]
pub struct KvRoutingCandidate {
    pub target: RoutableTarget,
    pub active_requests: u64,
    pub healthy: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvRoutingMode {
    Exact,
    /// Some healthy targets are untrusted. Cache credit is used only from
    /// trusted sources; the remaining targets compete by normalized load.
    Degraded,
    Fallback,
}

/// Structured decision evidence; contains no prompt text or token IDs.
#[derive(Debug, Clone)]
pub struct KvCandidateScore {
    pub target: RoutableTarget,
    pub credited_blocks: f64,
    pub normalized_load: f64,
    pub cost: f64,
}

#[derive(Debug, Clone)]
pub struct KvRoutingDecision {
    pub target: Option<RoutableTarget>,
    pub mode: KvRoutingMode,
    pub fallback_reason: Option<&'static str>,
    pub scores: Vec<KvCandidateScore>,
}

impl KvRoutingDecision {
    /// Serialize bounded per-candidate decision evidence, never request tokens
    /// or text. Instance/rank identities are supplied by the ingestion owner.
    pub fn to_json(&self) -> serde_json::Value {
        let target = |t: &RoutableTarget| {
            serde_json::json!({
                "instance_id": t.instance_id.as_ref(), "dp_rank": t.dp_rank
            })
        };
        serde_json::json!({
            "target": self.target.as_ref().map(target),
            "mode": match self.mode {
                KvRoutingMode::Exact => "exact",
                KvRoutingMode::Degraded => "degraded",
                KvRoutingMode::Fallback => "fallback"
            },
            "fallback_reason": self.fallback_reason,
            "scores": self.scores.iter().map(|s| serde_json::json!({
                "target": target(&s.target), "credited_blocks": s.credited_blocks,
                "normalized_load": s.normalized_load, "cost": s.cost
            })).collect::<Vec<_>>()
        })
    }
}

/// Configurable weighting of uncached-prefix fraction and normalized active
/// load. Transfer credit is penalty^hops; it is not a latency prediction.
#[derive(Debug, Clone)]
pub struct TrustedKvSelector {
    cache_weight: f64,
    transfer_penalty: f64,
    max_event_age: Duration,
}

impl TrustedKvSelector {
    pub fn new(
        cache_weight: f64,
        transfer_penalty: f64,
        max_event_age: Duration,
    ) -> Result<Self, &'static str> {
        if !cache_weight.is_finite()
            || !(0.0..=1.0).contains(&cache_weight)
            || !transfer_penalty.is_finite()
            || !(0.0..=1.0).contains(&transfer_penalty)
            || max_event_age.is_zero()
        {
            return Err("weights must be finite values in [0,1] and freshness must be positive");
        }
        Ok(Self {
            cache_weight,
            transfer_penalty,
            max_event_age,
        })
    }

    /// Query a model-scoped index and choose a concrete instance/DP rank.
    /// When every healthy candidate is trusted this is an exact decision. If
    /// only part of the pool is trusted, cache credit is restricted to that
    /// subset and the decision is marked degraded. Without any trusted source,
    /// selection uses deterministic least-load fallback and never queries the
    /// untrusted index.
    pub fn select(
        &self,
        provider: &dyn TieredMatchProvider,
        query: &MatchQuery,
        candidates: &[KvRoutingCandidate],
        trust: &KvTrustRegistry,
        now: Instant,
    ) -> KvRoutingDecision {
        let eligible: Vec<_> = candidates.iter().filter(|c| c.healthy).collect();
        let trust_failure = eligible
            .iter()
            .find_map(|c| trust.failure(&c.target, now, self.max_event_age));
        let trusted_count = eligible
            .iter()
            .filter(|c| trust.failure(&c.target, now, self.max_event_age).is_none())
            .count();
        let (mode, reason) = if eligible.is_empty() {
            (KvRoutingMode::Fallback, Some("no_healthy_targets"))
        } else if query.local_hashes.is_empty() {
            (KvRoutingMode::Fallback, Some("empty_query"))
        } else if trusted_count == 0 {
            (KvRoutingMode::Fallback, trust_failure)
        } else if trusted_count < eligible.len() {
            (KvRoutingMode::Degraded, trust_failure)
        } else {
            (KvRoutingMode::Exact, None)
        };
        let matches = if mode == KvRoutingMode::Fallback {
            Vec::new()
        } else {
            provider.find_tiered_matches(query)
        };
        let maximum_load = eligible
            .iter()
            .map(|c| c.active_requests)
            .max()
            .unwrap_or(0) as f64;
        let query_length = query.local_hashes.len() as f64;
        let mut scores: Vec<_> = eligible
            .iter()
            .map(|candidate| {
                let mut credited_blocks = 0.0_f64;
                let trusted = trust
                    .failure(&candidate.target, now, self.max_event_age)
                    .is_none();
                for hit in matches.iter().filter(|m| {
                    trusted
                        && m.target == candidate.target
                        && query.tiers_of_interest.contains(&m.tier)
                }) {
                    // Later held segments cannot be credited across a missing
                    // leading prefix. The multi-segment transfer scorer is later work.
                    let prefix = hit
                        .segments
                        .iter()
                        .filter(|(start, end)| *start == 0 && end > start)
                        .map(|(_, end)| f64::from(*end).min(query_length))
                        .fold(0.0_f64, f64::max);
                    let tier_hops = match hit.tier {
                        StorageTier::Device => 0,
                        StorageTier::HostPinned => 1,
                        StorageTier::Disk => 2,
                        StorageTier::External => 3,
                    };
                    let hops = tier_hops + i32::from(hit.locality == Locality::Remote);
                    credited_blocks =
                        credited_blocks.max(prefix * self.transfer_penalty.powi(hops));
                }
                let normalized_load = candidate.active_requests as f64 / (maximum_load + 1.0);
                let cost = if mode == KvRoutingMode::Fallback {
                    normalized_load
                } else {
                    self.cache_weight * (1.0 - credited_blocks / query_length)
                        + (1.0 - self.cache_weight) * normalized_load
                };
                KvCandidateScore {
                    target: candidate.target.clone(),
                    credited_blocks,
                    normalized_load,
                    cost,
                }
            })
            .collect();
        scores.sort_by(|a, b| {
            a.cost
                .total_cmp(&b.cost)
                .then_with(|| a.target.instance_id.cmp(&b.target.instance_id))
                .then_with(|| a.target.dp_rank.cmp(&b.target.dp_rank))
        });
        KvRoutingDecision {
            target: scores.first().map(|s| s.target.clone()),
            mode,
            fallback_reason: reason,
            scores,
        }
    }
}
