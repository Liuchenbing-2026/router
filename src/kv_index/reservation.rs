//! Paired P/D reservation contract with compensating cleanup.
//!
//! P/D engines may not expose one distributed transaction spanning prefill
//! retention, transfer identity and decode admission. This module therefore
//! implements the Router-side orchestration contract: reserve prefill first,
//! reserve decode second, and compensate the first leg if the second fails.
//! The backend must make reservations idempotent by routing attempt and stage,
//! and enforce the lease TTL so cancellation or an ambiguous network timeout
//! cannot leak a half-pair. An optional coordinator operation timeout bounds
//! each backend call; a timed-out reservation is reported as ambiguous because
//! the remote stage may already have accepted it.

use std::error::Error;
use std::fmt;
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use super::pairing::PdTargetPair;
use super::RoutableTarget;

/// The stage owning one reservation lease.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PdReservationStage {
    Prefill,
    Decode,
}

impl PdReservationStage {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
        }
    }
}

/// Correlation data for one paired reservation. Prompt text and token IDs are
/// intentionally absent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PdReservationIntent {
    pub pair: PdTargetPair,
    /// Stable identity for one Router P/D dispatch attempt. This may differ
    /// from a client-visible request ID and must change when a new routing
    /// attempt selects a different target pair.
    pub attempt_id: String,
    pub request_id: String,
    /// Connector-visible transfer identity. When unset, a connector may
    /// allocate one in the prefill lease; decode must inherit that value.
    pub transfer_id: Option<String>,
    /// Maximum time the engine may retain either half without an explicit
    /// release. Must be non-zero.
    pub lease_ttl: Duration,
}

/// Backend-specific handle proving that one engine stage accepted a lease.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PdReservationLease {
    pub stage: PdReservationStage,
    pub target: RoutableTarget,
    pub attempt_id: String,
    pub request_id: String,
    pub transfer_id: Option<String>,
    /// Decode leases must reference the exact prefill lease that authorized
    /// this admission. Prefill leases leave this unset.
    pub prefill_lease_id: Option<String>,
    /// Opaque backend handle. This value is not included in routing receipts.
    pub lease_id: String,
}

/// Result of attempting to compensate one reservation leg.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PdCleanupStatus {
    NotAcquired,
    /// The remote stage may have accepted the reservation even though Router
    /// did not receive a lease before the operation timeout.
    Ambiguous,
    Released,
    Failed(String),
}

impl fmt::Display for PdCleanupStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotAcquired => write!(f, "not_acquired"),
            Self::Ambiguous => write!(f, "ambiguous"),
            Self::Released => write!(f, "released"),
            Self::Failed(reason) => write!(f, "release_failed: {reason}"),
        }
    }
}

/// A paired reservation failed before a usable guard could be returned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PdReservationError {
    InvalidIntent(String),
    PrefillRejected {
        reason: String,
        cleanup: PdCleanupStatus,
    },
    DecodeRejected {
        reason: String,
        decode_cleanup: PdCleanupStatus,
        prefill_cleanup: PdCleanupStatus,
    },
}

impl fmt::Display for PdReservationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidIntent(reason) => write!(f, "invalid P/D reservation intent: {reason}"),
            Self::PrefillRejected { reason, cleanup } => {
                write!(
                    f,
                    "prefill reservation rejected: {reason}; cleanup={cleanup}"
                )
            }
            Self::DecodeRejected {
                reason,
                decode_cleanup,
                prefill_cleanup,
            } => write!(
                f,
                "decode reservation rejected: {reason}; decode_cleanup={decode_cleanup}; \
                 prefill_cleanup={prefill_cleanup}"
            ),
        }
    }
}

impl Error for PdReservationError {}

/// Engine/connector adapter used by the paired reservation coordinator.
///
/// Implementations must:
/// - key reservations by `(attempt_id, stage)` so retries are idempotent;
/// - use a fresh `attempt_id` whenever routing starts a new attempt, even if
///   the client request ID is reused;
/// - return a lease only after the stage has accepted the reservation;
/// - retain the transfer identity with the prefill lease and make decode
///   inherit a connector-generated identity;
/// - bind a decode lease to the exact prefill lease ID, not merely the attempt,
///   so a partially expired retry cannot mix a new prefill with stale decode;
/// - expire unreleased leases no later than `lease_ttl`; and
/// - make release safe to retry.
#[async_trait]
pub trait PdReservationBackend: Send + Sync + fmt::Debug {
    async fn reserve_prefill(
        &self,
        intent: &PdReservationIntent,
    ) -> Result<PdReservationLease, String>;

    async fn reserve_decode(
        &self,
        intent: &PdReservationIntent,
        prefill: &PdReservationLease,
    ) -> Result<PdReservationLease, String>;

    async fn release(&self, lease: &PdReservationLease) -> Result<(), String>;
}

/// Both leases produced by a successful paired reservation.
#[derive(Debug)]
pub struct PdReservationGuard {
    backend: Arc<dyn PdReservationBackend>,
    operation_timeout: Option<Duration>,
    prefill: PdReservationLease,
    decode: PdReservationLease,
}

impl PdReservationGuard {
    pub fn prefill(&self) -> &PdReservationLease {
        &self.prefill
    }

    pub fn decode(&self) -> &PdReservationLease {
        &self.decode
    }

    /// Release decode before prefill because decode admission depends on
    /// prefill retention. Both release calls are attempted even if one fails.
    /// The guard remains available so callers can retry an idempotent backend
    /// release after a partial failure.
    pub async fn release(&self) -> PdReleaseReport {
        let decode = release_status(
            self.backend.as_ref(),
            &self.decode,
            self.operation_timeout,
            "decode release",
        )
        .await;
        let prefill = release_status(
            self.backend.as_ref(),
            &self.prefill,
            self.operation_timeout,
            "prefill release",
        )
        .await;
        PdReleaseReport { prefill, decode }
    }

    /// Bounded routing evidence. Lease IDs and transfer identities stay out of
    /// receipts because they may be high-cardinality or connector-sensitive.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "prefill": target_json(&self.prefill.target),
            "decode": target_json(&self.decode.target),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PdReleaseReport {
    pub prefill: PdCleanupStatus,
    pub decode: PdCleanupStatus,
}

impl PdReleaseReport {
    pub fn all_released(&self) -> bool {
        self.prefill == PdCleanupStatus::Released && self.decode == PdCleanupStatus::Released
    }
}

/// Reserves both P/D legs and compensates the prefill leg if decode cannot be
/// admitted.
#[derive(Debug, Clone)]
pub struct PdReservationCoordinator {
    backend: Arc<dyn PdReservationBackend>,
    operation_timeout: Option<Duration>,
}

impl PdReservationCoordinator {
    pub fn new(backend: Arc<dyn PdReservationBackend>) -> Self {
        Self {
            backend,
            operation_timeout: None,
        }
    }

    /// Bound each backend reservation and release call. The timeout must be
    /// shorter than the request's lease TTL so an ambiguous remote acceptance
    /// can expire after Router returns.
    pub fn with_operation_timeout(
        backend: Arc<dyn PdReservationBackend>,
        operation_timeout: Duration,
    ) -> Result<Self, &'static str> {
        if operation_timeout.is_zero() {
            return Err("operation timeout must be non-zero");
        }
        Ok(Self {
            backend,
            operation_timeout: Some(operation_timeout),
        })
    }

    pub async fn reserve(
        &self,
        intent: PdReservationIntent,
    ) -> Result<PdReservationGuard, PdReservationError> {
        validate_intent(&intent)?;
        if self
            .operation_timeout
            .is_some_and(|timeout| timeout >= intent.lease_ttl)
        {
            return Err(PdReservationError::InvalidIntent(
                "operation timeout must be shorter than lease_ttl".to_string(),
            ));
        }

        let prefill = match bounded_backend_call(
            self.backend.reserve_prefill(&intent),
            self.operation_timeout,
            "prefill reservation",
        )
        .await
        {
            Ok(lease) => lease,
            Err(error) => {
                let cleanup = error.cleanup_if_unacquired();
                return Err(PdReservationError::PrefillRejected {
                    reason: error.reason,
                    cleanup,
                });
            }
        };

        if let Some(reason) = validate_lease(
            &prefill,
            PdReservationStage::Prefill,
            &intent.pair.prefill,
            &intent,
            intent.transfer_id.as_deref(),
            intent.transfer_id.is_some(),
            None,
        ) {
            let cleanup = release_status(
                self.backend.as_ref(),
                &prefill,
                self.operation_timeout,
                "prefill release",
            )
            .await;
            return Err(PdReservationError::PrefillRejected {
                reason: format!("invalid backend lease: {reason}"),
                cleanup,
            });
        }

        let decode = match bounded_backend_call(
            self.backend.reserve_decode(&intent, &prefill),
            self.operation_timeout,
            "decode reservation",
        )
        .await
        {
            Ok(lease) => lease,
            Err(error) => {
                let decode_cleanup = error.cleanup_if_unacquired();
                let prefill_cleanup = release_status(
                    self.backend.as_ref(),
                    &prefill,
                    self.operation_timeout,
                    "prefill release",
                )
                .await;
                return Err(PdReservationError::DecodeRejected {
                    reason: error.reason,
                    decode_cleanup,
                    prefill_cleanup,
                });
            }
        };

        if let Some(reason) = validate_lease(
            &decode,
            PdReservationStage::Decode,
            &intent.pair.decode,
            &intent,
            prefill.transfer_id.as_deref(),
            true,
            Some(prefill.lease_id.as_str()),
        ) {
            let decode_cleanup = release_status(
                self.backend.as_ref(),
                &decode,
                self.operation_timeout,
                "decode release",
            )
            .await;
            let prefill_cleanup = release_status(
                self.backend.as_ref(),
                &prefill,
                self.operation_timeout,
                "prefill release",
            )
            .await;
            return Err(PdReservationError::DecodeRejected {
                reason: format!("invalid backend lease: {reason}"),
                decode_cleanup,
                prefill_cleanup,
            });
        }

        Ok(PdReservationGuard {
            backend: Arc::clone(&self.backend),
            operation_timeout: self.operation_timeout,
            prefill,
            decode,
        })
    }
}

fn validate_intent(intent: &PdReservationIntent) -> Result<(), PdReservationError> {
    if intent.request_id.trim().is_empty() {
        return Err(PdReservationError::InvalidIntent(
            "request_id must not be empty".to_string(),
        ));
    }
    if intent.attempt_id.trim().is_empty() {
        return Err(PdReservationError::InvalidIntent(
            "attempt_id must not be empty".to_string(),
        ));
    }
    if intent
        .transfer_id
        .as_deref()
        .is_some_and(|value| value.trim().is_empty())
    {
        return Err(PdReservationError::InvalidIntent(
            "transfer_id must not be empty when supplied".to_string(),
        ));
    }
    if intent.lease_ttl.is_zero() {
        return Err(PdReservationError::InvalidIntent(
            "lease_ttl must be non-zero".to_string(),
        ));
    }
    Ok(())
}

fn validate_lease(
    lease: &PdReservationLease,
    stage: PdReservationStage,
    target: &RoutableTarget,
    intent: &PdReservationIntent,
    expected_transfer_id: Option<&str>,
    match_transfer_id: bool,
    expected_prefill_lease_id: Option<&str>,
) -> Option<&'static str> {
    if lease.stage != stage {
        return Some("stage mismatch");
    }
    if &lease.target != target {
        return Some("target mismatch");
    }
    if lease.attempt_id != intent.attempt_id {
        return Some("attempt_id mismatch");
    }
    if lease.request_id != intent.request_id {
        return Some("request_id mismatch");
    }
    if match_transfer_id && lease.transfer_id.as_deref() != expected_transfer_id {
        return Some("transfer_id mismatch");
    }
    if lease.prefill_lease_id.as_deref() != expected_prefill_lease_id {
        return Some("prefill_lease_id mismatch");
    }
    if lease.lease_id.trim().is_empty() {
        return Some("lease_id is empty");
    }
    None
}

async fn release_status(
    backend: &dyn PdReservationBackend,
    lease: &PdReservationLease,
    operation_timeout: Option<Duration>,
    operation: &'static str,
) -> PdCleanupStatus {
    match bounded_backend_call(backend.release(lease), operation_timeout, operation).await {
        Ok(()) => PdCleanupStatus::Released,
        Err(error) => PdCleanupStatus::Failed(error.reason),
    }
}

type BackendResult<T> = Result<T, BoundedBackendError>;

struct BoundedBackendError {
    reason: String,
    timed_out: bool,
}

impl BoundedBackendError {
    fn cleanup_if_unacquired(&self) -> PdCleanupStatus {
        if self.timed_out {
            PdCleanupStatus::Ambiguous
        } else {
            PdCleanupStatus::NotAcquired
        }
    }
}

async fn bounded_backend_call<T, F>(
    future: F,
    operation_timeout: Option<Duration>,
    operation: &'static str,
) -> BackendResult<T>
where
    F: Future<Output = Result<T, String>>,
{
    let Some(operation_timeout) = operation_timeout else {
        return future.await.map_err(|reason| BoundedBackendError {
            reason,
            timed_out: false,
        });
    };

    match tokio::time::timeout(operation_timeout, future).await {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(reason)) => Err(BoundedBackendError {
            reason,
            timed_out: false,
        }),
        Err(_) => Err(BoundedBackendError {
            reason: format!("{operation} timed out after {operation_timeout:?}"),
            timed_out: true,
        }),
    }
}

fn target_json(target: &RoutableTarget) -> serde_json::Value {
    serde_json::json!({
        "instance_id": target.instance_id.as_ref(),
        "dp_rank": target.dp_rank,
    })
}
