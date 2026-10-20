//! Atomic prefill/decode target pairing over stage-local trusted KV state.
//!
//! This is a selection boundary, not proof that an engine accepted a KV
//! transfer or retained its blocks. Callers must reserve and verify those
//! engine-side resources after receiving a pair.

use std::time::{Duration, Instant};

use super::routing::{
    KvRoutingCandidate, KvRoutingDecision, KvRoutingMode, KvTrustRegistry, TrustedKvSelector,
};
use super::{MatchQuery, RoutableTarget, TieredMatchProvider};

/// Inputs for one stage. Prefill and decode may use different indexes,
/// queries, candidate sets and trust registries.
pub struct PdStageSelection<'a> {
    pub provider: &'a dyn TieredMatchProvider,
    pub query: &'a MatchQuery,
    pub candidates: &'a [KvRoutingCandidate],
    pub trust: &'a KvTrustRegistry,
}

/// A concrete pair. It is created only when both stages selected a target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PdTargetPair {
    pub prefill: RoutableTarget,
    pub decode: RoutableTarget,
}

/// Two stage receipts plus the atomic pair result.
///
/// The aggregate mode is the least trusted stage mode. A fallback pair can
/// still contain concrete targets, but exact cache credit is never claimed for
/// the fallback stage.
#[derive(Debug, Clone)]
pub struct PdKvPairDecision {
    pub prefill: KvRoutingDecision,
    pub decode: KvRoutingDecision,
    pub pair: Option<PdTargetPair>,
    pub mode: KvRoutingMode,
    pub pair_reason: Option<&'static str>,
}

impl PdKvPairDecision {
    /// Structured route receipt with bounded evidence. Prompt text, token IDs
    /// and transfer identifiers are intentionally absent.
    pub fn to_json(&self) -> serde_json::Value {
        let target = |target: &RoutableTarget| {
            serde_json::json!({
                "instance_id": target.instance_id.as_ref(),
                "dp_rank": target.dp_rank,
            })
        };
        serde_json::json!({
            "mode": mode_name(self.mode),
            "pair_reason": self.pair_reason,
            "pair": self.pair.as_ref().map(|pair| serde_json::json!({
                "prefill": target(&pair.prefill),
                "decode": target(&pair.decode),
            })),
            "prefill": self.prefill.to_json(),
            "decode": self.decode.to_json(),
        })
    }
}

/// Selects a prefill/decode pair while keeping both stage evidence domains
/// isolated. An unavailable stage makes the whole pair unavailable.
#[derive(Debug, Clone)]
pub struct PdKvPairSelector {
    prefill: TrustedKvSelector,
    decode: TrustedKvSelector,
}

impl PdKvPairSelector {
    pub fn new(
        prefill_cache_weight: f64,
        prefill_transfer_penalty: f64,
        decode_cache_weight: f64,
        decode_transfer_penalty: f64,
        max_event_age: Duration,
    ) -> Result<Self, &'static str> {
        Ok(Self {
            prefill: TrustedKvSelector::new(
                prefill_cache_weight,
                prefill_transfer_penalty,
                max_event_age,
            )?,
            decode: TrustedKvSelector::new(
                decode_cache_weight,
                decode_transfer_penalty,
                max_event_age,
            )?,
        })
    }

    pub fn select(
        &self,
        prefill: PdStageSelection<'_>,
        decode: PdStageSelection<'_>,
        now: Instant,
    ) -> PdKvPairDecision {
        let prefill = self.prefill.select(
            prefill.provider,
            prefill.query,
            prefill.candidates,
            prefill.trust,
            now,
        );
        let decode = self.decode.select(
            decode.provider,
            decode.query,
            decode.candidates,
            decode.trust,
            now,
        );
        let pair = prefill
            .target
            .clone()
            .zip(decode.target.clone())
            .map(|(prefill, decode)| PdTargetPair { prefill, decode });
        let mode = aggregate_mode(&prefill, &decode, pair.is_some());
        let pair_reason = pair_reason(&prefill, &decode, pair.is_some(), mode);
        PdKvPairDecision {
            prefill,
            decode,
            pair,
            mode,
            pair_reason,
        }
    }
}

fn aggregate_mode(
    prefill: &KvRoutingDecision,
    decode: &KvRoutingDecision,
    has_pair: bool,
) -> KvRoutingMode {
    if !has_pair
        || prefill.mode == KvRoutingMode::Fallback
        || decode.mode == KvRoutingMode::Fallback
    {
        KvRoutingMode::Fallback
    } else if prefill.mode == KvRoutingMode::Degraded || decode.mode == KvRoutingMode::Degraded {
        KvRoutingMode::Degraded
    } else {
        KvRoutingMode::Exact
    }
}

fn pair_reason(
    prefill: &KvRoutingDecision,
    decode: &KvRoutingDecision,
    has_pair: bool,
    mode: KvRoutingMode,
) -> Option<&'static str> {
    if prefill.target.is_none() {
        return prefill.fallback_reason.or(Some("prefill_no_target"));
    }
    if decode.target.is_none() {
        return decode.fallback_reason.or(Some("decode_no_target"));
    }
    if !has_pair {
        return Some("pair_incomplete");
    }
    match mode {
        KvRoutingMode::Exact => None,
        KvRoutingMode::Degraded => prefill
            .fallback_reason
            .or(decode.fallback_reason)
            .or(Some("stage_degraded")),
        KvRoutingMode::Fallback => {
            if prefill.mode == KvRoutingMode::Fallback {
                prefill
                    .fallback_reason
                    .or(decode.fallback_reason)
                    .or(Some("stage_fallback"))
            } else {
                decode
                    .fallback_reason
                    .or(prefill.fallback_reason)
                    .or(Some("stage_fallback"))
            }
        }
    }
}

fn mode_name(mode: KvRoutingMode) -> &'static str {
    match mode {
        KvRoutingMode::Exact => "exact",
        KvRoutingMode::Degraded => "degraded",
        KvRoutingMode::Fallback => "fallback",
    }
}
