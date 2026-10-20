use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::Notify;
use tokio::time::{sleep, Instant};
use vllm_router_rs::kv_index::pairing::PdTargetPair;
use vllm_router_rs::kv_index::reservation::{
    PdCleanupStatus, PdReservationBackend, PdReservationCoordinator, PdReservationError,
    PdReservationIntent, PdReservationLease, PdReservationStage,
};
use vllm_router_rs::kv_index::RoutableTarget;

#[derive(Debug, Clone)]
struct LeaseRecord {
    lease: PdReservationLease,
    expires_at: Instant,
}

#[derive(Debug, Default)]
struct BackendState {
    prefill: HashMap<String, LeaseRecord>,
    decode: HashMap<String, LeaseRecord>,
    prefill_creations: usize,
    decode_creations: usize,
    decode_delay: Duration,
}

impl BackendState {
    fn sweep_expired(&mut self, now: Instant) -> (usize, usize) {
        self.prefill.retain(|_, record| record.expires_at > now);
        self.decode.retain(|_, record| record.expires_at > now);
        (self.prefill.len(), self.decode.len())
    }
}

/// Test-only reference adapter implementing the two properties the real
/// connector must provide: attempt-scoped idempotency and bounded TTL.
#[derive(Debug, Default)]
struct ReferenceBackend {
    state: Mutex<BackendState>,
    decode_started: Notify,
}

impl ReferenceBackend {
    fn lease(
        &self,
        intent: &PdReservationIntent,
        stage: PdReservationStage,
        prefill_lease_id: Option<&str>,
        generation: usize,
    ) -> PdReservationLease {
        let target = match stage {
            PdReservationStage::Prefill => intent.pair.prefill.clone(),
            PdReservationStage::Decode => intent.pair.decode.clone(),
        };
        PdReservationLease {
            stage,
            target,
            attempt_id: intent.attempt_id.clone(),
            request_id: intent.request_id.clone(),
            transfer_id: intent.transfer_id.clone(),
            prefill_lease_id: prefill_lease_id.map(str::to_string),
            lease_id: format!("{}-{}-{generation}", stage.as_str(), intent.attempt_id),
        }
    }

    fn live_leases(&self) -> (usize, usize) {
        self.state.lock().unwrap().sweep_expired(Instant::now())
    }

    fn creation_counts(&self) -> (usize, usize) {
        let state = self.state.lock().unwrap();
        (state.prefill_creations, state.decode_creations)
    }

    fn expire_prefill(&self, attempt_id: &str) {
        let mut state = self.state.lock().unwrap();
        let record = state.prefill.get_mut(attempt_id).unwrap();
        record.expires_at = Instant::now() - Duration::from_millis(1);
    }
}

#[async_trait]
impl PdReservationBackend for ReferenceBackend {
    async fn reserve_prefill(
        &self,
        intent: &PdReservationIntent,
    ) -> Result<PdReservationLease, String> {
        let now = Instant::now();
        let mut state = self.state.lock().unwrap();
        state.sweep_expired(now);
        if let Some(record) = state.prefill.get(&intent.attempt_id) {
            if record.lease.target != intent.pair.prefill
                || record.lease.request_id != intent.request_id
            {
                return Err("attempt_id is already bound to another prefill intent".to_string());
            }
            return Ok(record.lease.clone());
        }

        state.prefill_creations += 1;
        let mut lease = self.lease(
            intent,
            PdReservationStage::Prefill,
            None,
            state.prefill_creations,
        );
        if lease.transfer_id.is_none() {
            lease.transfer_id = Some(format!("connector-transfer-{}", intent.attempt_id));
        }
        state.prefill.insert(
            intent.attempt_id.clone(),
            LeaseRecord {
                lease: lease.clone(),
                expires_at: now + intent.lease_ttl,
            },
        );
        Ok(lease)
    }

    async fn reserve_decode(
        &self,
        intent: &PdReservationIntent,
        prefill: &PdReservationLease,
    ) -> Result<PdReservationLease, String> {
        let existing = {
            let now = Instant::now();
            let mut state = self.state.lock().unwrap();
            state.sweep_expired(now);
            state.decode.get(&intent.attempt_id).cloned()
        };
        if let Some(record) = existing {
            if record.lease.target != intent.pair.decode
                || record.lease.request_id != intent.request_id
                || record.lease.prefill_lease_id.as_deref() != Some(prefill.lease_id.as_str())
            {
                return Err(
                    "attempt_id is already bound to a different decode or prefill lease"
                        .to_string(),
                );
            }
            return Ok(record.lease);
        }

        self.decode_started.notify_one();
        let decode_delay = self.state.lock().unwrap().decode_delay;
        sleep(decode_delay).await;

        let now = Instant::now();
        let mut state = self.state.lock().unwrap();
        state.sweep_expired(now);
        if !state.prefill.contains_key(&intent.attempt_id) {
            return Err("prefill lease no longer exists".to_string());
        }
        state.decode_creations += 1;
        let mut lease = self.lease(
            intent,
            PdReservationStage::Decode,
            Some(prefill.lease_id.as_str()),
            state.decode_creations,
        );
        lease.transfer_id = prefill.transfer_id.clone();
        state.decode.insert(
            intent.attempt_id.clone(),
            LeaseRecord {
                lease: lease.clone(),
                expires_at: now + intent.lease_ttl,
            },
        );
        Ok(lease)
    }

    async fn release(&self, lease: &PdReservationLease) -> Result<(), String> {
        let now = Instant::now();
        let mut state = self.state.lock().unwrap();
        state.sweep_expired(now);
        let records = match lease.stage {
            PdReservationStage::Prefill => &mut state.prefill,
            PdReservationStage::Decode => &mut state.decode,
        };
        if let Some(record) = records.get(&lease.attempt_id) {
            if record.lease.lease_id != lease.lease_id {
                return Err("lease_id mismatch".to_string());
            }
            records.remove(&lease.attempt_id);
        }
        Ok(())
    }
}

fn target(instance: &str, rank: u32) -> RoutableTarget {
    RoutableTarget {
        instance_id: instance.into(),
        dp_rank: rank,
    }
}

fn intent(attempt_id: &str) -> PdReservationIntent {
    PdReservationIntent {
        pair: PdTargetPair {
            prefill: target("prefill-a", 3),
            decode: target("decode-b", 7),
        },
        attempt_id: attempt_id.to_string(),
        request_id: "client-request-reused-on-retry".to_string(),
        transfer_id: None,
        lease_ttl: Duration::from_millis(30),
    }
}

#[tokio::test]
async fn retries_are_idempotent_within_one_attempt() {
    let backend = Arc::new(ReferenceBackend::default());
    let coordinator = PdReservationCoordinator::new(backend.clone());

    let first = coordinator.reserve(intent("attempt-a")).await.unwrap();
    let retry = coordinator.reserve(intent("attempt-a")).await.unwrap();

    assert_eq!(first.prefill(), retry.prefill());
    assert_eq!(first.decode(), retry.decode());
    assert_eq!(first.prefill().transfer_id, first.decode().transfer_id);
    assert_eq!(backend.creation_counts(), (1, 1));

    assert!(first.release().await.all_released());
    assert!(retry.release().await.all_released());
    assert_eq!(backend.live_leases(), (0, 0));
}

#[tokio::test]
async fn reused_client_request_id_cannot_cross_attempt_leases() {
    let backend = Arc::new(ReferenceBackend::default());
    let coordinator = PdReservationCoordinator::new(backend.clone());

    let first = coordinator.reserve(intent("attempt-a")).await.unwrap();
    let second = coordinator.reserve(intent("attempt-b")).await.unwrap();

    assert_ne!(first.prefill().lease_id, second.prefill().lease_id);
    assert_ne!(first.decode().lease_id, second.decode().lease_id);
    assert_eq!(backend.creation_counts(), (2, 2));
}

#[tokio::test]
async fn canceled_coordination_relies_on_backend_lease_expiry() {
    let backend = Arc::new(ReferenceBackend::default());
    backend.state.lock().unwrap().decode_delay = Duration::from_secs(2);
    let coordinator = PdReservationCoordinator::new(backend.clone());

    let task = tokio::spawn(async move { coordinator.reserve(intent("attempt-canceled")).await });
    backend.decode_started.notified().await;
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());

    assert_eq!(backend.live_leases(), (1, 0));
    sleep(Duration::from_millis(40)).await;
    assert_eq!(backend.live_leases(), (0, 0));
}

#[tokio::test]
async fn partial_prefill_expiry_cannot_mix_with_stale_decode() {
    let backend = Arc::new(ReferenceBackend::default());
    let coordinator = PdReservationCoordinator::new(backend.clone());
    let mut first_intent = intent("attempt-partial");
    first_intent.transfer_id = Some("fixed-transfer".to_string());
    let first = coordinator.reserve(first_intent.clone()).await.unwrap();

    backend.expire_prefill("attempt-partial");
    let error = coordinator.reserve(first_intent).await.unwrap_err();
    assert!(matches!(
        error,
        PdReservationError::DecodeRejected {
            decode_cleanup: PdCleanupStatus::NotAcquired,
            prefill_cleanup: PdCleanupStatus::Released,
            ref reason,
        } if reason.contains("different decode or prefill lease")
    ));
    assert_eq!(backend.creation_counts(), (2, 1));
    assert_eq!(backend.live_leases(), (0, 1));

    assert!(first.release().await.all_released());
    assert_eq!(backend.live_leases(), (0, 0));
}
