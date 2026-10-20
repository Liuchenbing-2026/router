use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use tokio::time::sleep;
use vllm_router_rs::kv_index::pairing::PdTargetPair;
use vllm_router_rs::kv_index::reservation::{
    PdCleanupStatus, PdReservationBackend, PdReservationCoordinator, PdReservationError,
    PdReservationIntent, PdReservationLease, PdReservationStage,
};
use vllm_router_rs::kv_index::RoutableTarget;

#[derive(Debug, Default)]
struct BackendState {
    calls: Vec<String>,
    prefill_error: Option<String>,
    decode_error: Option<String>,
    prefill_release_error: Option<String>,
    decode_release_error: Option<String>,
    corrupt_prefill_lease: bool,
    corrupt_decode_lease: bool,
    corrupt_decode_prefill_binding: bool,
    generated_transfer_id: Option<String>,
    prefill_delay_ms: u64,
    decode_delay_ms: u64,
    prefill_release_delay_ms: u64,
    decode_release_delay_ms: u64,
}

#[derive(Debug, Default)]
struct MockBackend {
    state: Mutex<BackendState>,
}

impl MockBackend {
    fn with_state(f: impl FnOnce(&mut BackendState)) -> Arc<Self> {
        let backend = Arc::new(Self::default());
        f(&mut backend.state.lock().unwrap());
        backend
    }

    fn calls(&self) -> Vec<String> {
        self.state.lock().unwrap().calls.clone()
    }

    fn lease(
        &self,
        intent: &PdReservationIntent,
        stage: PdReservationStage,
        corrupt: bool,
        prefill_lease_id: Option<&str>,
    ) -> PdReservationLease {
        let target = match stage {
            PdReservationStage::Prefill => {
                if corrupt {
                    target("wrong", 99)
                } else {
                    intent.pair.prefill.clone()
                }
            }
            PdReservationStage::Decode => {
                if corrupt {
                    target("wrong", 99)
                } else {
                    intent.pair.decode.clone()
                }
            }
        };
        PdReservationLease {
            stage,
            target,
            attempt_id: intent.attempt_id.clone(),
            request_id: intent.request_id.clone(),
            transfer_id: intent.transfer_id.clone(),
            prefill_lease_id: prefill_lease_id.map(str::to_string),
            lease_id: format!("{}-lease", stage.as_str()),
        }
    }
}

#[async_trait]
impl PdReservationBackend for MockBackend {
    async fn reserve_prefill(
        &self,
        intent: &PdReservationIntent,
    ) -> Result<PdReservationLease, String> {
        let (error, corrupt, delay) = {
            let mut state = self.state.lock().unwrap();
            state
                .calls
                .push(format!("reserve-prefill:{}", intent.request_id));
            (
                state.prefill_error.clone(),
                state.corrupt_prefill_lease,
                Duration::from_millis(state.prefill_delay_ms),
            )
        };
        if let Some(error) = error {
            return Err(error);
        }
        if !delay.is_zero() {
            sleep(delay).await;
        }
        let mut lease = self.lease(intent, PdReservationStage::Prefill, corrupt, None);
        if lease.transfer_id.is_none() {
            lease.transfer_id = self.state.lock().unwrap().generated_transfer_id.clone();
        }
        Ok(lease)
    }

    async fn reserve_decode(
        &self,
        intent: &PdReservationIntent,
        prefill: &PdReservationLease,
    ) -> Result<PdReservationLease, String> {
        let (error, corrupt, delay) = {
            let mut state = self.state.lock().unwrap();
            state.calls.push(format!(
                "reserve-decode:{}:{}",
                intent.request_id, prefill.lease_id
            ));
            (
                state.decode_error.clone(),
                state.corrupt_decode_lease,
                Duration::from_millis(state.decode_delay_ms),
            )
        };
        if let Some(error) = error {
            return Err(error);
        }
        if !delay.is_zero() {
            sleep(delay).await;
        }
        let corrupt_binding = self.state.lock().unwrap().corrupt_decode_prefill_binding;
        let binding = if corrupt_binding {
            Some("stale-prefill-lease")
        } else {
            Some(prefill.lease_id.as_str())
        };
        let mut lease = self.lease(intent, PdReservationStage::Decode, corrupt, binding);
        lease.transfer_id = prefill.transfer_id.clone();
        Ok(lease)
    }

    async fn release(&self, lease: &PdReservationLease) -> Result<(), String> {
        let (error, delay) = {
            let mut state = self.state.lock().unwrap();
            state.calls.push(format!(
                "release-{}:{}",
                lease.stage.as_str(),
                lease.lease_id
            ));
            let error = match lease.stage {
                PdReservationStage::Prefill => state.prefill_release_error.clone(),
                PdReservationStage::Decode => state.decode_release_error.clone(),
            };
            let delay = match lease.stage {
                PdReservationStage::Prefill => state.prefill_release_delay_ms,
                PdReservationStage::Decode => state.decode_release_delay_ms,
            };
            (error, Duration::from_millis(delay))
        };
        if !delay.is_zero() {
            sleep(delay).await;
        }
        match error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

fn target(instance: &str, rank: u32) -> RoutableTarget {
    RoutableTarget {
        instance_id: instance.into(),
        dp_rank: rank,
    }
}

fn intent() -> PdReservationIntent {
    PdReservationIntent {
        pair: PdTargetPair {
            prefill: target("prefill-a", 3),
            decode: target("decode-b", 7),
        },
        attempt_id: "attempt-42".to_string(),
        request_id: "request-42".to_string(),
        transfer_id: Some("transfer-42".to_string()),
        lease_ttl: Duration::from_secs(30),
    }
}

#[tokio::test]
async fn successful_reservation_returns_both_validated_leases() {
    let backend = Arc::new(MockBackend::default());
    let coordinator = PdReservationCoordinator::new(backend.clone());

    let guard = coordinator.reserve(intent()).await.unwrap();
    assert_eq!(guard.prefill().stage, PdReservationStage::Prefill);
    assert_eq!(guard.prefill().target, target("prefill-a", 3));
    assert!(guard.prefill().prefill_lease_id.is_none());
    assert_eq!(guard.decode().stage, PdReservationStage::Decode);
    assert_eq!(guard.decode().target, target("decode-b", 7));
    assert_eq!(
        guard.decode().prefill_lease_id.as_deref(),
        Some("prefill-lease")
    );
    assert_eq!(
        backend.calls(),
        [
            "reserve-prefill:request-42",
            "reserve-decode:request-42:prefill-lease"
        ]
    );
}

#[tokio::test]
async fn prefill_generated_transfer_identity_flows_into_decode() {
    let backend = MockBackend::with_state(|state| {
        state.generated_transfer_id = Some("connector-transfer-42".to_string());
    });
    let coordinator = PdReservationCoordinator::new(backend);
    let mut request = intent();
    request.transfer_id = None;

    let guard = coordinator.reserve(request).await.unwrap();
    assert_eq!(
        guard.prefill().transfer_id.as_deref(),
        Some("connector-transfer-42")
    );
    assert_eq!(guard.decode().transfer_id, guard.prefill().transfer_id);
}

#[tokio::test]
async fn prefill_timeout_is_ambiguous_instead_of_no_lease() {
    let backend = MockBackend::with_state(|state| state.prefill_delay_ms = 1_000);
    let coordinator = PdReservationCoordinator::with_operation_timeout(
        backend.clone(),
        Duration::from_millis(10),
    )
    .unwrap();

    let error = coordinator.reserve(intent()).await.unwrap_err();
    assert!(matches!(
        error,
        PdReservationError::PrefillRejected {
            cleanup: PdCleanupStatus::Ambiguous,
            ..
        }
    ));
    assert_eq!(backend.calls(), ["reserve-prefill:request-42"]);
}

#[tokio::test]
async fn decode_timeout_rolls_back_prefill_and_marks_decode_ambiguous() {
    let backend = MockBackend::with_state(|state| state.decode_delay_ms = 1_000);
    let coordinator = PdReservationCoordinator::with_operation_timeout(
        backend.clone(),
        Duration::from_millis(10),
    )
    .unwrap();

    let error = coordinator.reserve(intent()).await.unwrap_err();
    assert!(matches!(
        error,
        PdReservationError::DecodeRejected {
            decode_cleanup: PdCleanupStatus::Ambiguous,
            prefill_cleanup: PdCleanupStatus::Released,
            ..
        }
    ));
    assert_eq!(
        backend.calls(),
        [
            "reserve-prefill:request-42",
            "reserve-decode:request-42:prefill-lease",
            "release-prefill:prefill-lease"
        ]
    );
}

#[tokio::test]
async fn release_timeout_is_reported_and_remains_retryable() {
    let backend = MockBackend::with_state(|state| state.decode_release_delay_ms = 1_000);
    let coordinator = PdReservationCoordinator::with_operation_timeout(
        backend.clone(),
        Duration::from_millis(10),
    )
    .unwrap();
    let guard = coordinator.reserve(intent()).await.unwrap();

    let first = guard.release().await;
    assert!(matches!(
        first.decode,
        PdCleanupStatus::Failed(ref reason) if reason.contains("decode release timed out")
    ));
    assert_eq!(first.prefill, PdCleanupStatus::Released);

    backend.state.lock().unwrap().decode_release_delay_ms = 0;
    let retry = guard.release().await;
    assert!(retry.all_released());
}

#[tokio::test]
async fn operation_timeout_must_be_nonzero_and_shorter_than_lease_ttl() {
    let backend = Arc::new(MockBackend::default());
    assert_eq!(
        PdReservationCoordinator::with_operation_timeout(backend.clone(), Duration::ZERO)
            .unwrap_err(),
        "operation timeout must be non-zero"
    );

    let coordinator =
        PdReservationCoordinator::with_operation_timeout(backend.clone(), Duration::from_secs(30))
            .unwrap();
    let error = coordinator.reserve(intent()).await.unwrap_err();
    assert!(matches!(error, PdReservationError::InvalidIntent(_)));
    assert!(backend.calls().is_empty());
}

#[tokio::test]
async fn prefill_rejection_never_attempts_decode() {
    let backend = MockBackend::with_state(|state| {
        state.prefill_error = Some("prefill p3 saturated".to_string());
    });
    let coordinator = PdReservationCoordinator::new(backend.clone());

    let error = coordinator.reserve(intent()).await.unwrap_err();
    assert!(matches!(
        error,
        PdReservationError::PrefillRejected {
            cleanup: PdCleanupStatus::NotAcquired,
            ..
        }
    ));
    assert_eq!(backend.calls(), ["reserve-prefill:request-42"]);
}

#[tokio::test]
async fn decode_rejection_compensates_prefill() {
    let backend = MockBackend::with_state(|state| {
        state.decode_error = Some("decode d7 admission rejected".to_string());
    });
    let coordinator = PdReservationCoordinator::new(backend.clone());

    let error = coordinator.reserve(intent()).await.unwrap_err();
    assert!(matches!(
        error,
        PdReservationError::DecodeRejected {
            decode_cleanup: PdCleanupStatus::NotAcquired,
            prefill_cleanup: PdCleanupStatus::Released,
            ..
        }
    ));
    assert_eq!(
        backend.calls(),
        [
            "reserve-prefill:request-42",
            "reserve-decode:request-42:prefill-lease",
            "release-prefill:prefill-lease"
        ]
    );
}

#[tokio::test]
async fn failed_prefill_compensation_is_reported_and_ttl_remains_required() {
    let backend = MockBackend::with_state(|state| {
        state.decode_error = Some("decode unavailable".to_string());
        state.prefill_release_error = Some("release endpoint timed out".to_string());
    });
    let coordinator = PdReservationCoordinator::new(backend);

    let error = coordinator.reserve(intent()).await.unwrap_err();
    assert!(matches!(
        error,
        PdReservationError::DecodeRejected {
            prefill_cleanup: PdCleanupStatus::Failed(ref reason),
            ..
        } if reason == "release endpoint timed out"
    ));
}

#[tokio::test]
async fn invalid_prefill_lease_is_released_before_returning_error() {
    let backend = MockBackend::with_state(|state| state.corrupt_prefill_lease = true);
    let coordinator = PdReservationCoordinator::new(backend.clone());

    let error = coordinator.reserve(intent()).await.unwrap_err();
    assert!(matches!(
        error,
        PdReservationError::PrefillRejected {
            cleanup: PdCleanupStatus::Released,
            ..
        }
    ));
    assert_eq!(
        backend.calls(),
        [
            "reserve-prefill:request-42",
            "release-prefill:prefill-lease"
        ]
    );
}

#[tokio::test]
async fn invalid_decode_lease_releases_both_legs() {
    let backend = MockBackend::with_state(|state| state.corrupt_decode_lease = true);
    let coordinator = PdReservationCoordinator::new(backend.clone());

    let error = coordinator.reserve(intent()).await.unwrap_err();
    assert!(matches!(
        error,
        PdReservationError::DecodeRejected {
            decode_cleanup: PdCleanupStatus::Released,
            prefill_cleanup: PdCleanupStatus::Released,
            ..
        }
    ));
    assert_eq!(
        backend.calls(),
        [
            "reserve-prefill:request-42",
            "reserve-decode:request-42:prefill-lease",
            "release-decode:decode-lease",
            "release-prefill:prefill-lease"
        ]
    );
}

#[tokio::test]
async fn decode_lease_must_bind_to_the_reserved_prefill_lease() {
    let backend = MockBackend::with_state(|state| {
        state.corrupt_decode_prefill_binding = true;
    });
    let coordinator = PdReservationCoordinator::new(backend.clone());

    let error = coordinator.reserve(intent()).await.unwrap_err();
    assert!(matches!(
        error,
        PdReservationError::DecodeRejected {
            decode_cleanup: PdCleanupStatus::Released,
            prefill_cleanup: PdCleanupStatus::Released,
            ref reason,
        } if reason.contains("prefill_lease_id mismatch")
    ));
    assert_eq!(
        backend.calls(),
        [
            "reserve-prefill:request-42",
            "reserve-decode:request-42:prefill-lease",
            "release-decode:decode-lease",
            "release-prefill:prefill-lease"
        ]
    );
}

#[tokio::test]
async fn explicit_release_attempts_both_legs_even_when_one_fails() {
    let backend = MockBackend::with_state(|state| {
        state.decode_release_error = Some("decode release failed".to_string());
    });
    let coordinator = PdReservationCoordinator::new(backend.clone());
    let guard = coordinator.reserve(intent()).await.unwrap();

    let report = guard.release().await;
    assert_eq!(
        report.decode,
        PdCleanupStatus::Failed("decode release failed".to_string())
    );
    assert_eq!(report.prefill, PdCleanupStatus::Released);
    assert!(!report.all_released());
    assert_eq!(
        backend.calls(),
        [
            "reserve-prefill:request-42",
            "reserve-decode:request-42:prefill-lease",
            "release-decode:decode-lease",
            "release-prefill:prefill-lease"
        ]
    );
}

#[tokio::test]
async fn failed_release_can_be_retried_with_the_same_guard() {
    let backend = MockBackend::with_state(|state| {
        state.decode_release_error = Some("decode release failed".to_string());
    });
    let coordinator = PdReservationCoordinator::new(backend.clone());
    let guard = coordinator.reserve(intent()).await.unwrap();

    let first = guard.release().await;
    assert!(!first.all_released());

    backend.state.lock().unwrap().decode_release_error = None;
    let retry = guard.release().await;
    assert!(retry.all_released());
    assert_eq!(
        backend.calls(),
        [
            "reserve-prefill:request-42",
            "reserve-decode:request-42:prefill-lease",
            "release-decode:decode-lease",
            "release-prefill:prefill-lease",
            "release-decode:decode-lease",
            "release-prefill:prefill-lease"
        ]
    );
}

#[tokio::test]
async fn invalid_intents_do_not_touch_backend() {
    let backend = Arc::new(MockBackend::default());
    let coordinator = PdReservationCoordinator::new(backend.clone());
    let mut empty_request = intent();
    empty_request.request_id = "  ".to_string();
    let mut empty_attempt = intent();
    empty_attempt.attempt_id = " ".to_string();
    let mut empty_transfer = intent();
    empty_transfer.transfer_id = Some(" ".to_string());
    let mut zero_ttl = intent();
    zero_ttl.lease_ttl = Duration::ZERO;

    for invalid in [empty_request, empty_attempt, empty_transfer, zero_ttl] {
        let error = coordinator.reserve(invalid).await.unwrap_err();
        assert!(matches!(error, PdReservationError::InvalidIntent(_)));
    }
    assert!(backend.calls().is_empty());
}

#[tokio::test]
async fn guard_receipt_contains_targets_but_no_identities_or_prompt_data() {
    let backend = Arc::new(MockBackend::default());
    let coordinator = PdReservationCoordinator::new(backend);
    let guard = coordinator.reserve(intent()).await.unwrap();

    let receipt = guard.to_json();
    assert_eq!(receipt["prefill"]["instance_id"], "prefill-a");
    assert_eq!(receipt["prefill"]["dp_rank"], 3);
    assert_eq!(receipt["decode"]["instance_id"], "decode-b");
    assert_eq!(receipt["decode"]["dp_rank"], 7);
    assert!(receipt.get("attempt_id").is_none());
    assert!(receipt.get("lease_id").is_none());
    assert!(receipt.get("transfer_id").is_none());
    assert!(receipt.get("request_id").is_none());
    assert!(receipt.get("prompt").is_none());
}
