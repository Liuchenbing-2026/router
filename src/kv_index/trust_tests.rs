use super::*;

fn t(ms: u64) -> Duration {
    Duration::from_millis(ms)
}

fn key(rank: u32) -> TrustKey {
    TrustKey {
        namespace: "model-a/sha256/block16".into(),
        instance_id: "worker-a".into(),
        dp_rank: rank,
    }
}

fn arbiter() -> TrustArbiter {
    TrustArbiter::new(t(100)).unwrap()
}

fn bootstrap(a: &mut TrustArbiter, k: TrustKey) -> SubscriptionLease {
    let lease = a.subscribe(k, 1, t(0)).unwrap();
    let ticket = a.ticket(&lease, t(0)).unwrap();
    a.recover(&ticket, RecoveryProof::Snapshot { through: 100 }, t(0))
        .unwrap();
    lease
}

#[test]
fn startup_and_live_events_do_not_prove_bootstrap() {
    let mut a = arbiter();
    assert_eq!(a.mode(&key(0), t(0)), Ok(TrustMode::Fallback));
    let l = a.subscribe(key(0), 1, t(0)).unwrap();
    assert_eq!(a.mode(&key(0), t(0)), Ok(TrustMode::Fallback));
    a.advance(&l, 50, t(1)).unwrap();
    a.advance(&l, 51, t(2)).unwrap();
    assert_eq!(a.mode(&key(0), t(2)), Ok(TrustMode::Degraded));
    let p = a.ticket(&l, t(2)).unwrap();
    assert_eq!(
        a.recover(
            &p,
            RecoveryProof::Replay {
                from: 0,
                through: 51
            },
            t(2)
        ),
        Err(TrustError::ReplayNeedsCheckpoint)
    );
}

#[test]
fn contiguous_events_preserve_exact_but_gap_latches() {
    let mut a = arbiter();
    let l = bootstrap(&mut a, key(0));
    a.advance(&l, 101, t(1)).unwrap();
    assert_eq!(a.mode(&key(0), t(1)), Ok(TrustMode::Exact));
    a.advance(&l, 104, t(2)).unwrap();
    a.advance(&l, 105, t(3)).unwrap();
    assert_eq!(a.mode(&key(0), t(3)), Ok(TrustMode::Degraded));
}

#[test]
fn only_replay_covering_all_missing_history_recovers() {
    let mut a = arbiter();
    let l = bootstrap(&mut a, key(0));
    a.advance(&l, 103, t(1)).unwrap();
    a.advance(&l, 108, t(2)).unwrap();
    let p = a.ticket(&l, t(2)).unwrap();
    for (from, through) in [(104, 108), (101, 107)] {
        assert_eq!(
            a.recover(&p, RecoveryProof::Replay { from, through }, t(3)),
            Err(TrustError::InsufficientCoverage)
        );
    }
    a.recover(
        &p,
        RecoveryProof::Replay {
            from: 101,
            through: 108,
        },
        t(4),
    )
    .unwrap();
    assert_eq!(a.mode(&key(0), t(4)), Ok(TrustMode::Exact));
    assert_eq!(
        a.recover(&p, RecoveryProof::Snapshot { through: 108 }, t(4)),
        Err(TrustError::StaleProof)
    );
}

#[test]
fn stale_stream_cannot_self_heal_without_a_mode_poll() {
    let mut a = arbiter();
    let l = bootstrap(&mut a, key(0));
    a.advance(&l, 101, t(100)).unwrap();
    assert_eq!(a.mode(&key(0), t(100)), Ok(TrustMode::Degraded));
}

#[test]
fn duplicate_and_out_of_order_events_do_not_refresh_freshness() {
    let mut a = arbiter();
    let l = bootstrap(&mut a, key(0));
    assert!(!a.advance(&l, 100, t(80)).unwrap());
    assert!(!a.advance(&l, 99, t(99)).unwrap());
    assert_eq!(a.mode(&key(0), t(99)), Ok(TrustMode::Exact));
    assert_eq!(a.mode(&key(0), t(100)), Ok(TrustMode::Degraded));
}

#[test]
fn rank_instance_and_namespace_are_isolated() {
    let mut a = arbiter();
    let mut other_instance = key(0);
    other_instance.instance_id = "worker-b".into();
    let mut other_model = key(0);
    other_model.namespace = "model-b/sha256/block16".into();
    let l = bootstrap(&mut a, key(0));
    for k in [key(1), other_instance.clone(), other_model.clone()] {
        bootstrap(&mut a, k);
    }
    a.invalidate(&l, t(1)).unwrap();
    assert_eq!(a.mode(&key(0), t(1)), Ok(TrustMode::Degraded));
    for k in [key(1), other_instance, other_model] {
        assert_eq!(a.mode(&k, t(1)), Ok(TrustMode::Exact));
    }
}

#[test]
fn reconnect_fences_old_events_tickets_and_retirement() {
    let mut a = arbiter();
    let old = bootstrap(&mut a, key(0));
    let p = a.ticket(&old, t(1)).unwrap();
    let new = a.subscribe(key(0), 1, t(2)).unwrap();
    assert_ne!(old, new);
    assert_eq!(a.advance(&old, 101, t(3)), Err(TrustError::UnknownLease));
    assert_eq!(a.retire(&old), Err(TrustError::UnknownLease));
    assert_eq!(
        a.recover(&p, RecoveryProof::Snapshot { through: 100 }, t(3)),
        Err(TrustError::UnknownLease)
    );
    assert_eq!(a.mode(&key(0), t(3)), Ok(TrustMode::Fallback));
}

#[test]
fn incarnation_restart_requires_new_bootstrap() {
    let mut a = arbiter();
    let old = bootstrap(&mut a, key(0));
    let new = a.subscribe(key(0), 2, t(1)).unwrap();
    assert_eq!(
        a.subscribe(key(0), 1, t(2)),
        Err(TrustError::StaleIncarnation)
    );
    assert_eq!(a.advance(&old, 101, t(2)), Err(TrustError::UnknownLease));
    a.advance(&new, 0, t(2)).unwrap();
    assert_eq!(a.mode(&key(0), t(2)), Ok(TrustMode::Degraded));
    let p = a.ticket(&new, t(2)).unwrap();
    a.recover(&p, RecoveryProof::ClearBarrier { through: 0 }, t(3))
        .unwrap();
    assert_eq!(a.mode(&key(0), t(3)), Ok(TrustMode::Exact));
}

#[test]
fn retire_releases_entry_but_never_reuses_generation() {
    let mut a = arbiter();
    let old = bootstrap(&mut a, key(0));
    a.retire(&old).unwrap();
    assert!(a.entries.is_empty());
    assert_eq!(a.mode(&key(0), t(1)), Ok(TrustMode::Fallback));
    let new = a.subscribe(key(0), 1, t(2)).unwrap();
    assert_ne!(old, new);
    assert_eq!(a.advance(&old, 101, t(3)), Err(TrustError::UnknownLease));
}

#[test]
fn events_and_explicit_faults_fence_recovery_tickets() {
    for fault in [false, true] {
        let mut a = arbiter();
        let l = bootstrap(&mut a, key(0));
        a.invalidate(&l, t(1)).unwrap();
        let p = a.ticket(&l, t(1)).unwrap();
        if fault {
            a.invalidate(&l, t(2)).unwrap();
        } else {
            a.advance(&l, 101, t(2)).unwrap();
        }
        assert_eq!(
            a.recover(&p, RecoveryProof::Snapshot { through: 101 }, t(3)),
            Err(TrustError::StaleProof)
        );
    }
}

#[test]
fn delayed_proof_is_rejected_at_exact_ttl_boundary() {
    let mut a = arbiter();
    let l = a.subscribe(key(0), 1, t(0)).unwrap();
    let p = a.ticket(&l, t(0)).unwrap();
    assert_eq!(
        a.recover(&p, RecoveryProof::Snapshot { through: 0 }, t(100)),
        Err(TrustError::StaleProof)
    );
    assert_eq!(a.mode(&key(0), t(100)), Ok(TrustMode::Fallback));
}

#[test]
fn proof_cannot_rollback_high_water_or_invert_interval() {
    let mut a = arbiter();
    let l = bootstrap(&mut a, key(0));
    a.advance(&l, 103, t(1)).unwrap();
    let p = a.ticket(&l, t(1)).unwrap();
    for proof in [
        RecoveryProof::Snapshot { through: 99 },
        RecoveryProof::ClearBarrier { through: 99 },
    ] {
        assert_eq!(
            a.recover(&p, proof, t(2)),
            Err(TrustError::InsufficientCoverage)
        );
    }
    assert_eq!(
        a.recover(
            &p,
            RecoveryProof::Replay {
                from: 110,
                through: 103
            },
            t(2)
        ),
        Err(TrustError::InvalidProof)
    );
}

#[test]
fn monotonic_clock_regression_is_explicit() {
    let mut a = arbiter();
    let l = bootstrap(&mut a, key(0));
    let p = a.ticket(&l, t(10)).unwrap();
    assert_eq!(a.mode(&key(0), t(9)), Err(TrustError::ClockRegression));
    assert_eq!(a.advance(&l, 101, t(9)), Err(TrustError::ClockRegression));
    assert_eq!(
        a.recover(&p, RecoveryProof::Snapshot { through: 100 }, t(9)),
        Err(TrustError::ClockRegression)
    );
}

#[test]
fn sequence_max_does_not_overflow() {
    let mut a = arbiter();
    let l = a.subscribe(key(0), 1, t(0)).unwrap();
    let p = a.ticket(&l, t(0)).unwrap();
    a.recover(&p, RecoveryProof::Snapshot { through: u64::MAX }, t(0))
        .unwrap();
    assert!(!a.advance(&l, u64::MAX, t(1)).unwrap());
    a.invalidate(&l, t(1)).unwrap();
    let p = a.ticket(&l, t(1)).unwrap();
    assert_eq!(
        a.recover(
            &p,
            RecoveryProof::Replay {
                from: 0,
                through: u64::MAX
            },
            t(1)
        ),
        Err(TrustError::ReplayNeedsCheckpoint)
    );
}

#[test]
fn zero_ttl_and_counter_exhaustion_fail_closed() {
    assert!(matches!(
        TrustArbiter::new(t(0)),
        Err(TrustError::InvalidConfiguration)
    ));
    let mut a = arbiter();
    let l = bootstrap(&mut a, key(0));
    a.generation = u64::MAX;
    assert_eq!(
        a.subscribe(key(1), 1, t(1)),
        Err(TrustError::CounterExhausted)
    );
    a.entries.get_mut(&key(0)).unwrap().revision = u64::MAX;
    assert_eq!(a.advance(&l, 101, t(1)), Err(TrustError::CounterExhausted));
    assert_eq!(a.mode(&key(0), t(1)), Err(TrustError::CounterExhausted));
    assert!(!a.entries[&key(0)].exact);
}

#[test]
fn separate_arbiters_cannot_accept_each_others_leases_or_proofs() {
    let mut a = arbiter();
    let mut b = arbiter();
    let la = a.subscribe(key(0), 1, t(0)).unwrap();
    let lb = b.subscribe(key(0), 1, t(0)).unwrap();
    let proof_a = a.ticket(&la, t(0)).unwrap();
    assert_ne!(la, lb);
    assert_eq!(b.advance(&la, 1, t(1)), Err(TrustError::UnknownLease));
    assert_eq!(
        b.recover(&proof_a, RecoveryProof::Snapshot { through: 1 }, t(1)),
        Err(TrustError::UnknownLease)
    );
}

#[test]
fn synthetic_four_instances_sixteen_ranks_stay_independent() {
    let mut a = arbiter();
    let mut leases = Vec::new();
    for instance in 0..4 {
        for rank in 0..16 {
            let mut k = key(rank);
            k.instance_id = format!("worker-{instance}");
            leases.push(bootstrap(&mut a, k));
        }
    }
    a.invalidate(&leases[19], t(1)).unwrap();
    for (i, l) in leases.iter().enumerate() {
        assert_eq!(
            a.mode(&l.key, t(1)),
            Ok(if i == 19 {
                TrustMode::Degraded
            } else {
                TrustMode::Exact
            })
        );
    }
}
