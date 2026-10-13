//! Experimental, single-process trust contract. NOT wired into request routing.
//!
//! One arbiter is owned by a serialized supervisor. A namespace must identify the
//! complete index compatibility domain (model, hash mode, block size, etc.).
//! `RecoveryProof` asserts that a trusted producer has already verified AND
//! applied the indicated coverage to the index; this module cannot verify that
//! assertion. In particular, an ingestion high-water mark is NOT a proof.
//!
//! Times are durations from one monotonic process origin, never wall-clock time.
//! Call `mode` at consumption time: a previously returned `Exact` is not a lease.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct TrustKey {
    pub namespace: String,
    pub instance_id: String,
    pub dp_rank: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrustMode {
    Exact,
    Degraded,
    Fallback,
}

/// Opaque within a process; never deserialize or accept this from a request.
#[derive(Clone, Debug)]
pub struct SubscriptionLease {
    key: TrustKey,
    generation: u64,
    incarnation: u64,
    // Allocation identity prevents equal local counters in separate arbiters
    // from accepting each other's leases. Clones retain the same identity.
    fence: Arc<()>,
}

impl PartialEq for SubscriptionLease {
    fn eq(&self, other: &Self) -> bool {
        self.generation == other.generation
            && self.incarnation == other.incarnation
            && self.key == other.key
            && Arc::ptr_eq(&self.fence, &other.fence)
    }
}

impl Eq for SubscriptionLease {}

/// Captured before applying recovery under the supervisor's ordering contract.
/// Any intervening applied event/invalidation makes this ticket stale.
#[derive(Clone, Debug)]
pub struct RecoveryTicket {
    lease: SubscriptionLease,
    revision: u64,
    issued_at: Duration,
}

/// All sequence bounds are inclusive and refer to successfully applied events.
/// Snapshot/clear assertions require an index-wide barrier in this exact scope.
#[derive(Clone, Copy, Debug)]
pub enum RecoveryProof {
    Snapshot { through: u64 },
    ClearBarrier { through: u64 },
    Replay { from: u64, through: u64 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrustError {
    InvalidConfiguration,
    UnknownLease,
    StaleIncarnation,
    ClockRegression,
    StaleProof,
    InvalidProof,
    InsufficientCoverage,
    ReplayNeedsCheckpoint,
    CounterExhausted,
}

#[derive(Debug)]
struct Entry {
    lease: SubscriptionLease,
    revision: u64,
    high: Option<u64>,
    last_seen: Option<Duration>,
    last_clock: Duration,
    exact: bool,
    // First event after the last trusted checkpoint. Subsequent gaps never
    // replace it, so repairing only the most recent hole cannot restore trust.
    replay_from: Option<u64>,
    exhausted: bool,
}

impl Entry {
    fn bump(&mut self) -> Result<(), TrustError> {
        let Some(next) = self.revision.checked_add(1) else {
            self.exact = false;
            self.exhausted = true;
            return Err(TrustError::CounterExhausted);
        };
        self.revision = next;
        Ok(())
    }

    fn invalidate(&mut self) -> Result<(), TrustError> {
        if self.exact {
            self.replay_from = self.high.and_then(|n| n.checked_add(1));
        }
        self.exact = false;
        self.bump()
    }

    fn prepare(&mut self, now: Duration, max_age: Duration) -> Result<(), TrustError> {
        if self.exhausted {
            return Err(TrustError::CounterExhausted);
        }
        if now < self.last_clock {
            return Err(TrustError::ClockRegression);
        }
        self.last_clock = now;
        if self.exact && self.last_seen.is_some_and(|seen| now - seen >= max_age) {
            self.invalidate()?;
        }
        Ok(())
    }
}

/// Prototype only: no network, index mutation, policy selection or global HA.
/// Entries follow active subscriptions; retirement releases their memory.
#[derive(Debug)]
pub struct TrustArbiter {
    entries: HashMap<TrustKey, Entry>,
    generation: u64,
    max_age: Duration,
}

impl TrustArbiter {
    pub fn new(max_age: Duration) -> Result<Self, TrustError> {
        if max_age.is_zero() {
            return Err(TrustError::InvalidConfiguration);
        }
        Ok(Self {
            entries: HashMap::new(),
            generation: 0,
            max_age,
        })
    }

    /// The authoritative lifecycle owner supplies the incarnation and has
    /// already retired/fenced any old subscriber and its index mutations.
    /// Reconnecting the same incarnation still creates a fresh local fence.
    pub fn subscribe(
        &mut self,
        key: TrustKey,
        incarnation: u64,
        now: Duration,
    ) -> Result<SubscriptionLease, TrustError> {
        if let Some(old) = self.entries.get(&key) {
            if incarnation < old.lease.incarnation {
                return Err(TrustError::StaleIncarnation);
            }
            if now < old.last_clock {
                return Err(TrustError::ClockRegression);
            }
        }
        self.generation = self
            .generation
            .checked_add(1)
            .ok_or(TrustError::CounterExhausted)?;
        let lease = SubscriptionLease {
            key: key.clone(),
            generation: self.generation,
            incarnation,
            fence: Arc::new(()),
        };
        self.entries.insert(
            key,
            Entry {
                lease: lease.clone(),
                revision: 0,
                high: None,
                last_seen: None,
                last_clock: now,
                exact: false,
                replay_from: None,
                exhausted: false,
            },
        );
        Ok(lease)
    }

    fn entry(&mut self, lease: &SubscriptionLease) -> Result<&mut Entry, TrustError> {
        self.entries
            .get_mut(&lease.key)
            .filter(|entry| entry.lease == *lease)
            .ok_or(TrustError::UnknownLease)
    }

    /// Retiring one rank does not invalidate other ranks or shared cache owners.
    /// The lifecycle owner remains responsible for actual index cleanup.
    pub fn retire(&mut self, lease: &SubscriptionLease) -> Result<(), TrustError> {
        self.entry(lease)?;
        self.entries.remove(&lease.key);
        Ok(())
    }

    /// Accept a strictly decoded, successfully applied event. Returns false for
    /// duplicates/out-of-order delivery, which must not refresh freshness.
    /// A sequence reset is an explicit lifecycle operation, not an Advance.
    pub fn advance(
        &mut self,
        lease: &SubscriptionLease,
        seq: u64,
        now: Duration,
    ) -> Result<bool, TrustError> {
        let max_age = self.max_age;
        let entry = self.entry(lease)?;
        // Expire BEFORE recording a fresh event: otherwise a long outage could
        // silently self-heal without any caller having polled `mode` meanwhile.
        entry.prepare(now, max_age)?;
        if let Some(high) = entry.high {
            if seq <= high {
                return Ok(false);
            }
            if high.checked_add(1) != Some(seq) {
                entry.invalidate()?;
            }
        }
        entry.bump()?;
        entry.high = Some(seq);
        entry.last_seen = Some(now);
        Ok(true)
    }

    /// Explicit gap/decode failure/suspect-stream notification. No high-water
    /// mark is advanced. A second invalidation fences already-issued proofs.
    pub fn invalidate(
        &mut self,
        lease: &SubscriptionLease,
        now: Duration,
    ) -> Result<(), TrustError> {
        let max_age = self.max_age;
        let entry = self.entry(lease)?;
        entry.prepare(now, max_age)?;
        entry.invalidate()
    }

    pub fn ticket(
        &mut self,
        lease: &SubscriptionLease,
        now: Duration,
    ) -> Result<RecoveryTicket, TrustError> {
        let max_age = self.max_age;
        let entry = self.entry(lease)?;
        entry.prepare(now, max_age)?;
        Ok(RecoveryTicket {
            lease: lease.clone(),
            revision: entry.revision,
            issued_at: now,
        })
    }

    /// Called only after proof validation AND index application. Callers must
    /// serialize this with index updates; this object is NOT an index lock.
    pub fn recover(
        &mut self,
        ticket: &RecoveryTicket,
        proof: RecoveryProof,
        now: Duration,
    ) -> Result<(), TrustError> {
        let max_age = self.max_age;
        let entry = self.entry(&ticket.lease)?;
        entry.prepare(now, max_age)?;
        if entry.revision != ticket.revision || now - ticket.issued_at >= max_age {
            return Err(TrustError::StaleProof);
        }
        let through = match proof {
            RecoveryProof::Snapshot { through } | RecoveryProof::ClearBarrier { through } => {
                through
            }
            RecoveryProof::Replay { from, through } => {
                if from > through {
                    return Err(TrustError::InvalidProof);
                }
                let required = entry.replay_from.ok_or(TrustError::ReplayNeedsCheckpoint)?;
                if from > required {
                    return Err(TrustError::InsufficientCoverage);
                }
                through
            }
        };
        if entry.high.is_some_and(|high| through < high) {
            return Err(TrustError::InsufficientCoverage);
        }
        entry.bump()?;
        entry.high = Some(through);
        entry.last_seen = Some(now);
        entry.replay_from = None;
        entry.exact = true;
        Ok(())
    }

    /// Unknown/retired targets are fallback; partial or stale sources are
    /// degraded. Errors must be treated as zero exact credit by future callers.
    /// This is a PER-TARGET mode, not the aggregate per-model fallback decision.
    pub fn mode(&mut self, key: &TrustKey, now: Duration) -> Result<TrustMode, TrustError> {
        let Some(entry) = self.entries.get_mut(key) else {
            return Ok(TrustMode::Fallback);
        };
        entry.prepare(now, self.max_age)?;
        Ok(if entry.exact {
            TrustMode::Exact
        } else if entry.high.is_some() {
            TrustMode::Degraded
        } else {
            TrustMode::Fallback
        })
    }
}

#[cfg(test)]
#[path = "trust_tests.rs"]
mod tests;
