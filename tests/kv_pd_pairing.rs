use std::sync::Arc;
use std::time::{Duration, Instant};
use vllm_router_rs::kv_index::pairing::{PdKvPairSelector, PdStageSelection};
use vllm_router_rs::kv_index::routing::{KvRoutingCandidate, KvRoutingMode, KvTrustRegistry};
use vllm_router_rs::kv_index::{
    KvBlockIndexer, MatchQuery, ResidencyOwner, RoutableTarget, StorageTier,
};

fn target(instance: &str, rank: u32) -> RoutableTarget {
    RoutableTarget {
        instance_id: Arc::from(instance),
        dp_rank: rank,
    }
}

fn owner(instance: &str, rank: u32) -> ResidencyOwner {
    ResidencyOwner::Worker {
        source: Arc::from(instance),
        dp_rank: rank,
        incarnation: 0,
    }
}

fn query() -> MatchQuery {
    MatchQuery {
        group_idx: 0,
        local_hashes: vec![Arc::from("a"), Arc::from("b")],
        tiers_of_interest: vec![StorageTier::Device],
    }
}

fn candidates(instance: &str, ranks: &[u32]) -> Vec<KvRoutingCandidate> {
    ranks
        .iter()
        .map(|rank| KvRoutingCandidate {
            target: target(instance, *rank),
            active_requests: u64::from(*rank),
            healthy: true,
        })
        .collect()
}

fn trust(now: Instant, instance: &str, ranks: &[u32]) -> KvTrustRegistry {
    let mut trust = KvTrustRegistry::new(ranks.len().max(1)).unwrap();
    for rank in ranks {
        let target = target(instance, *rank);
        assert!(trust.register(target.clone(), 0, now));
        assert!(trust.reconcile(&target, 0, 1, now));
    }
    trust
}

fn selector() -> PdKvPairSelector {
    PdKvPairSelector::new(0.8, 0.5, 0.8, 0.5, Duration::from_secs(10)).unwrap()
}

#[test]
fn exact_stage_decisions_form_one_concrete_pair() {
    let now = Instant::now();
    let prefill_index = KvBlockIndexer::new();
    prefill_index.store(
        0,
        owner("p", 1),
        StorageTier::Device,
        None,
        &[
            (Arc::from("p-a"), Arc::from("a")),
            (Arc::from("p-b"), Arc::from("b")),
        ],
    );
    let decode_index = KvBlockIndexer::new();
    decode_index.store(
        0,
        owner("d", 1),
        StorageTier::Device,
        None,
        &[
            (Arc::from("d-a"), Arc::from("a")),
            (Arc::from("d-b"), Arc::from("b")),
        ],
    );
    let prefill_candidates = candidates("p", &[0, 1]);
    let decode_candidates = candidates("d", &[0, 1]);
    let prefill_trust = trust(now, "p", &[0, 1]);
    let decode_trust = trust(now, "d", &[0, 1]);

    let decision = selector().select(
        PdStageSelection {
            provider: &prefill_index,
            query: &query(),
            candidates: &prefill_candidates,
            trust: &prefill_trust,
        },
        PdStageSelection {
            provider: &decode_index,
            query: &query(),
            candidates: &decode_candidates,
            trust: &decode_trust,
        },
        now,
    );

    assert_eq!(decision.mode, KvRoutingMode::Exact);
    assert_eq!(decision.pair_reason, None);
    let pair = decision.pair.unwrap();
    assert_eq!(pair.prefill, target("p", 1));
    assert_eq!(pair.decode, target("d", 1));
}

#[test]
fn one_unavailable_stage_makes_the_whole_pair_incomplete() {
    let now = Instant::now();
    let prefill_index = KvBlockIndexer::new();
    let decode_index = KvBlockIndexer::new();
    let prefill_candidates = candidates("p", &[0]);
    let decode_candidates = Vec::new();
    let prefill_trust = trust(now, "p", &[0]);
    let decode_trust = KvTrustRegistry::new(1).unwrap();

    let decision = selector().select(
        PdStageSelection {
            provider: &prefill_index,
            query: &query(),
            candidates: &prefill_candidates,
            trust: &prefill_trust,
        },
        PdStageSelection {
            provider: &decode_index,
            query: &query(),
            candidates: &decode_candidates,
            trust: &decode_trust,
        },
        now,
    );

    assert_eq!(decision.prefill.mode, KvRoutingMode::Exact);
    assert_eq!(decision.decode.mode, KvRoutingMode::Fallback);
    assert_eq!(decision.mode, KvRoutingMode::Fallback);
    assert!(decision.pair.is_none());
    assert_eq!(decision.pair_reason, Some("no_healthy_targets"));
}

#[test]
fn trusted_fallback_preserves_a_concrete_pair_without_exact_credit() {
    let now = Instant::now();
    let prefill_index = KvBlockIndexer::new();
    let decode_index = KvBlockIndexer::new();
    let prefill_candidates = candidates("p", &[0]);
    let decode_candidates = candidates("d", &[0]);
    let prefill_trust = trust(now, "p", &[0]);
    let mut decode_trust = KvTrustRegistry::new(1).unwrap();
    assert!(decode_trust.register(target("d", 0), 0, now));

    let decision = selector().select(
        PdStageSelection {
            provider: &prefill_index,
            query: &query(),
            candidates: &prefill_candidates,
            trust: &prefill_trust,
        },
        PdStageSelection {
            provider: &decode_index,
            query: &query(),
            candidates: &decode_candidates,
            trust: &decode_trust,
        },
        now,
    );

    assert_eq!(decision.prefill.mode, KvRoutingMode::Exact);
    assert_eq!(decision.decode.mode, KvRoutingMode::Fallback);
    assert_eq!(decision.mode, KvRoutingMode::Fallback);
    assert_eq!(decision.pair_reason, Some("unreconciled"));
    assert_eq!(
        decision.pair.unwrap(),
        vllm_router_rs::kv_index::pairing::PdTargetPair {
            prefill: target("p", 0),
            decode: target("d", 0),
        }
    );
}

#[test]
fn one_degraded_stage_marks_the_pair_degraded() {
    let now = Instant::now();
    let prefill_index = KvBlockIndexer::new();
    let decode_index = KvBlockIndexer::new();
    let prefill_candidates = candidates("p", &[0]);
    let decode_candidates = candidates("d", &[0, 1]);
    let prefill_trust = trust(now, "p", &[0]);
    let mut decode_trust = trust(now, "d", &[0, 1]);
    decode_trust.mark_unavailable(&target("d", 1));

    let decision = selector().select(
        PdStageSelection {
            provider: &prefill_index,
            query: &query(),
            candidates: &prefill_candidates,
            trust: &prefill_trust,
        },
        PdStageSelection {
            provider: &decode_index,
            query: &query(),
            candidates: &decode_candidates,
            trust: &decode_trust,
        },
        now,
    );

    assert_eq!(decision.mode, KvRoutingMode::Degraded);
    assert_eq!(decision.pair_reason, Some("unavailable"));
    assert!(decision.pair.is_some());
}

#[test]
fn pair_receipt_contains_both_stage_evidence_but_no_request_payload() {
    let now = Instant::now();
    let prefill_index = KvBlockIndexer::new();
    let decode_index = KvBlockIndexer::new();
    let prefill_candidates = candidates("p", &[0]);
    let decode_candidates = candidates("d", &[0]);
    let prefill_trust = trust(now, "p", &[0]);
    let decode_trust = trust(now, "d", &[0]);

    let receipt = selector()
        .select(
            PdStageSelection {
                provider: &prefill_index,
                query: &query(),
                candidates: &prefill_candidates,
                trust: &prefill_trust,
            },
            PdStageSelection {
                provider: &decode_index,
                query: &query(),
                candidates: &decode_candidates,
                trust: &decode_trust,
            },
            now,
        )
        .to_json();

    assert_eq!(receipt["mode"], "exact");
    assert_eq!(receipt["pair"]["prefill"]["instance_id"], "p");
    assert_eq!(receipt["pair"]["decode"]["instance_id"], "d");
    assert!(receipt["prefill"]["scores"].is_array());
    assert!(receipt["decode"]["scores"].is_array());
    assert!(receipt.get("token_ids").is_none());
    assert!(receipt.get("prompt").is_none());
    assert!(receipt.get("transfer_id").is_none());
}
