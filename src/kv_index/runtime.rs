//! Fenced ingestion and selection for normal-dense, worker-owned device KV.
//!
//! A validated AllBlocksCleared or complete replay through the live watermark
//! is the completeness barrier. Live deltas alone never establish completeness.
//! Request metadata cannot publish residency.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;
use serde_json::Value;

use super::routing::{
    KvRecoveryState, KvRoutingCandidate, KvRoutingDecision, KvRoutingMode, KvTrustRegistry,
    TrustedKvSelector,
};
use super::snapshot::{
    validate_portable_snapshot, PortableCacheIdentity, PortableKvSnapshot, PortableSnapshotError,
    PortableSnapshotFence, PortableSnapshotFenceAuthority, PortableSnapshotImport,
    PortableSnapshotImportStatus, ValidatedPortableSource,
};
use super::{
    decode_batch, local_hashes, validate_device_batch, CacheKey, ClearScope, HashMode, KVEvent,
    KVEventBatch, KvBlockIndexer, KvEventSourcesResponse, MatchQuery, ResidencyOwner,
    RoutableTarget, StorageTier,
};
use crate::config::types::TrustedKvAwareConfig;
use crate::core::Worker;

struct Cursor {
    sequence: Option<u64>,
    incarnation: u64,
    applied_snapshot_id: Option<String>,
    applied_snapshot_checksum: Option<String>,
    snapshot_source_sha256: Option<String>,
    snapshot_created_at_unix_ms: Option<u64>,
    snapshot_observed_at_unix_ms: Option<u64>,
}

impl Cursor {
    fn new(incarnation: u64) -> Self {
        Self {
            sequence: None,
            incarnation,
            applied_snapshot_id: None,
            applied_snapshot_checksum: None,
            snapshot_source_sha256: None,
            snapshot_created_at_unix_ms: None,
            snapshot_observed_at_unix_ms: None,
        }
    }

    fn reset_incarnation(&mut self, incarnation: u64) {
        self.incarnation = incarnation;
        self.sequence = None;
        self.applied_snapshot_id = None;
        self.applied_snapshot_checksum = None;
        self.snapshot_source_sha256 = None;
        self.snapshot_created_at_unix_ms = None;
        self.snapshot_observed_at_unix_ms = None;
    }
}

struct Pool {
    id: u64,
    revision: u64,
    index: KvBlockIndexer,
    trust: KvTrustRegistry,
    cursors: HashMap<RoutableTarget, Cursor>,
    portable_identity: Option<PortableCacheIdentity>,
    snapshot_checksums: HashMap<String, String>,
    // A conservative allocation budget: removals do not refund historical
    // insertions because alias/topology metadata may outlive residency.
    announced: usize,
}

impl Pool {
    fn touch(&mut self) {
        self.revision = self.revision.wrapping_add(1);
    }
}

struct WorkerSource {
    generation: u64,
    key: Option<CacheKey>,
    instance: String,
    models: Vec<String>,
    ranks: Vec<u32>,
    endpoints: Vec<String>,
    tasks: Vec<tokio::task::AbortHandle>,
}

#[derive(Debug)]
struct PublisherSource {
    rank: u32,
    endpoint: String,
    topic: String,
    replay_endpoint: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SequenceAction {
    Apply,
    Duplicate,
    Replay { from: u64, through: u64 },
    Drop,
}

struct ReplayContext {
    key: CacheKey,
    incarnation: u64,
}

impl Drop for WorkerSource {
    fn drop(&mut self) {
        for task in &self.tasks {
            task.abort();
        }
    }
}

#[derive(Default)]
struct State {
    generation: u64,
    next_pool_id: u64,
    fence_revision: u64,
    workers: HashMap<String, WorkerSource>,
    pools: HashMap<CacheKey, Pool>,
    announced: usize,
    applied_snapshot_fence: Option<PortableSnapshotFence>,
}

impl State {
    fn allocate_pool_id(&mut self) -> Option<u64> {
        self.next_pool_id = self.next_pool_id.checked_add(1)?;
        Some(self.next_pool_id)
    }
}

pub struct KvRuntime {
    config: TrustedKvAwareConfig,
    state: Mutex<State>,
    context: Arc<zmq::Context>,
    snapshot_fence_authority: Option<Arc<dyn PortableSnapshotFenceAuthority>>,
}

impl std::fmt::Debug for KvRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KvRuntime")
            .field("workers", &self.state.lock().workers.len())
            .finish()
    }
}

impl KvRuntime {
    pub fn new(config: TrustedKvAwareConfig) -> Self {
        Self::with_snapshot_fence_authority(config, None)
    }

    /// Build a runtime with an optional durable fence authority. Supplying an
    /// authority makes stale-epoch rejection survive construction of a fresh
    /// runtime, provided the adapter itself persists the authoritative fence.
    pub fn with_snapshot_fence_authority(
        config: TrustedKvAwareConfig,
        snapshot_fence_authority: Option<Arc<dyn PortableSnapshotFenceAuthority>>,
    ) -> Self {
        Self {
            config,
            state: Mutex::new(State::default()),
            context: Arc::new(zmq::Context::new()),
            snapshot_fence_authority,
        }
    }

    /// Atomically replace one cache identity's local baseline with a complete
    /// portable snapshot. `expected_identity` is supplied by the trusted
    /// ingestion owner; the snapshot cannot nominate its own tokenizer.
    /// `expected_fence` is supplied by the trusted control-plane integration;
    /// the snapshot cannot authorize its own epoch.
    ///
    /// Validation and staging complete before any live state is swapped. A
    /// snapshot that omits a currently discovered target is rejected, because
    /// accepting it would mix old residency into the new baseline. The fence
    /// authority is called without holding the live state lock. If live
    /// ingestion, discovery or removal changes the target pool while that call
    /// is in flight, another import advances the global fence, or the target
    /// pool is deleted and recreated, this returns
    /// `ConcurrentSnapshotMutation`; retrying the same snapshot and fence is
    /// safe for an idempotent authority. Traffic in other cache pools does not
    /// force a retry.
    pub fn import_portable_snapshot(
        &self,
        snapshot: &PortableKvSnapshot,
        expected_identity: &PortableCacheIdentity,
        expected_fence: &PortableSnapshotFence,
    ) -> Result<PortableSnapshotImport, PortableSnapshotError> {
        let now_unix_ms = unix_millis()?;
        let validated = validate_portable_snapshot(
            snapshot,
            expected_identity,
            expected_fence,
            self.config.max_sources,
            self.config.max_blocks,
            now_unix_ms,
        )?;
        let (
            key,
            staged_pool_id,
            staged_pool_revision,
            staged_fence_revision,
            staged_index,
            staged_trust,
            staged_cursors,
        ) = {
            let mut state = self.state.lock();
            let existing_announced = state.announced;
            let matching_keys: Vec<_> = state
                .pools
                .keys()
                .filter(|key| {
                    key.model.as_ref() == expected_identity.model
                        && expected_identity.hash_mode.matches(key.hash_mode)
                        && key.block_size == expected_identity.block_size
                })
                .cloned()
                .collect();
            let [key] = matching_keys.as_slice() else {
                return Err(PortableSnapshotError::IdentityMismatch);
            };
            let key = key.clone();
            let applied_fence = state.applied_snapshot_fence.clone();
            let pool = state
                .pools
                .get_mut(&key)
                .ok_or(PortableSnapshotError::IdentityMismatch)?;
            if pool
                .portable_identity
                .as_ref()
                .is_some_and(|identity| identity != expected_identity)
            {
                return Err(PortableSnapshotError::IdentityMismatch);
            }

            let current_targets: HashSet<_> = pool.cursors.keys().cloned().collect();
            let snapshot_targets: HashSet<_> = validated
                .sources
                .iter()
                .map(|source| source.target.clone())
                .collect();
            if current_targets != snapshot_targets {
                return Err(PortableSnapshotError::SourceSetMismatch);
            }

            if pool
                .snapshot_checksums
                .get(&validated.snapshot_id)
                .is_some_and(|checksum| checksum != &validated.checksum_sha256)
            {
                return Err(PortableSnapshotError::SnapshotIdConflict);
            }

            if let Some(applied) = &applied_fence {
                if validated.fence.epoch < applied.epoch {
                    return Err(PortableSnapshotError::StaleEpoch);
                }
                if validated.fence.epoch == applied.epoch
                    && validated.fence.owner_id != applied.owner_id
                {
                    return Err(PortableSnapshotError::EpochOwnerConflict);
                }
            }

            let now = Instant::now();
            let max_age = Duration::from_secs(self.config.max_event_age_secs);
            let already_applied = validated.sources.iter().all(|source| {
                pool.cursors.get(&source.target).is_some_and(|cursor| {
                    cursor.applied_snapshot_id.as_deref() == Some(&validated.snapshot_id)
                        && cursor.applied_snapshot_checksum.as_deref()
                            == Some(&validated.checksum_sha256)
                        && cursor.sequence == Some(source.sequence)
                        && cursor.incarnation == source.incarnation
                        && pool
                            .trust
                            .status(&source.target, now, max_age)
                            .is_some_and(|status| status.state == KvRecoveryState::Healthy)
                })
            });
            if already_applied {
                return Ok(PortableSnapshotImport {
                    snapshot_id: validated.snapshot_id,
                    status: PortableSnapshotImportStatus::AlreadyApplied,
                    sources: validated.sources.len(),
                    blocks: validated.block_count,
                });
            }

            for source in &validated.sources {
                let cursor = pool
                    .cursors
                    .get(&source.target)
                    .ok_or(PortableSnapshotError::SourceSetMismatch)?;
                if source.incarnation < cursor.incarnation {
                    return Err(PortableSnapshotError::ObsoleteIncarnation);
                }
                if source.incarnation == cursor.incarnation
                    && cursor.sequence == Some(source.sequence)
                    && cursor
                        .snapshot_source_sha256
                        .as_ref()
                        .is_some_and(|content| content != &source.content_sha256)
                {
                    return Err(PortableSnapshotError::SnapshotConflict);
                }
                if source.incarnation == cursor.incarnation
                    && (cursor
                        .sequence
                        .is_some_and(|sequence| source.sequence < sequence)
                        || cursor
                            .snapshot_created_at_unix_ms
                            .is_some_and(|created| validated.created_at_unix_ms < created)
                        || cursor
                            .snapshot_observed_at_unix_ms
                            .is_some_and(|observed| source.observed_at_unix_ms < observed))
                {
                    return Err(PortableSnapshotError::Rollback);
                }
            }

            let retained_blocks = existing_announced.saturating_sub(pool.announced);
            if validated.block_count > self.config.max_blocks.saturating_sub(retained_blocks) {
                return Err(PortableSnapshotError::BlockBudgetExceeded);
            }

            let staged_index = KvBlockIndexer::new();
            let mut staged_trust = KvTrustRegistry::new(self.config.max_sources)
                .expect("validated snapshot source capacity");
            let mut staged_cursors = HashMap::with_capacity(validated.sources.len());
            for source in &validated.sources {
                stage_snapshot_source(
                    &staged_index,
                    source,
                    now_unix_ms,
                    now,
                    self.config.max_event_age_secs,
                )?;
                if !staged_trust.register(source.target.clone(), source.incarnation, now) {
                    return Err(PortableSnapshotError::Invalid(
                        "snapshot trust capacity exhausted",
                    ));
                }
                let observed_at = snapshot_observed_instant(
                    source.observed_at_unix_ms,
                    now_unix_ms,
                    now,
                    self.config.max_event_age_secs,
                )?;
                if !staged_trust.reconcile(
                    &source.target,
                    source.incarnation,
                    source.sequence,
                    observed_at,
                ) {
                    return Err(PortableSnapshotError::Invalid(
                        "snapshot source could not be reconciled",
                    ));
                }
                staged_cursors.insert(
                    source.target.clone(),
                    Cursor {
                        sequence: Some(source.sequence),
                        incarnation: source.incarnation,
                        applied_snapshot_id: Some(validated.snapshot_id.clone()),
                        applied_snapshot_checksum: Some(validated.checksum_sha256.clone()),
                        snapshot_source_sha256: Some(source.content_sha256.clone()),
                        snapshot_created_at_unix_ms: Some(validated.created_at_unix_ms),
                        snapshot_observed_at_unix_ms: Some(source.observed_at_unix_ms),
                    },
                );
            }
            (
                key,
                pool.id,
                pool.revision,
                state.fence_revision,
                staged_index,
                staged_trust,
                staged_cursors,
            )
        };

        if let Some(authority) = &self.snapshot_fence_authority {
            authority.admit_and_advance(&validated.fence)?;
        }

        let mut state = self.state.lock();
        if state.fence_revision != staged_fence_revision {
            return Err(PortableSnapshotError::ConcurrentSnapshotMutation);
        }
        let current_announced = state.announced;
        let pool = state
            .pools
            .get_mut(&key)
            .filter(|pool| pool.id == staged_pool_id && pool.revision == staged_pool_revision)
            .ok_or(PortableSnapshotError::ConcurrentSnapshotMutation)?;
        let old_announced = pool.announced;
        let new_announced = current_announced
            .saturating_sub(old_announced)
            .saturating_add(validated.block_count);
        if new_announced > self.config.max_blocks {
            return Err(PortableSnapshotError::BlockBudgetExceeded);
        }
        let mut snapshot_checksums = std::mem::take(&mut pool.snapshot_checksums);
        snapshot_checksums.insert(
            validated.snapshot_id.clone(),
            validated.checksum_sha256.clone(),
        );
        pool.index = staged_index;
        pool.trust = staged_trust;
        pool.cursors = staged_cursors;
        pool.portable_identity = Some(expected_identity.clone());
        pool.snapshot_checksums = snapshot_checksums;
        pool.announced = validated.block_count;
        pool.touch();
        state.announced = new_announced;
        state.applied_snapshot_fence = Some(validated.fence);
        state.fence_revision = state.fence_revision.wrapping_add(1);
        Ok(PortableSnapshotImport {
            snapshot_id: validated.snapshot_id,
            status: PortableSnapshotImportStatus::Applied,
            sources: validated.sources.len(),
            blocks: validated.block_count,
        })
    }

    /// Removal invalidates the discovery lease before stopping subscribers.
    /// An old HTTP discovery response or queued event cannot reattach a worker.
    pub fn remove_worker(&self, url: &str) {
        let mut state = self.state.lock();
        Self::remove_locked(&mut state, url);
    }

    fn remove_locked(state: &mut State, url: &str) {
        let Some(worker) = state.workers.remove(url) else {
            return;
        };
        if let Some(key) = &worker.key {
            let mut empty = false;
            if let Some(pool) = state.pools.get_mut(key) {
                pool.touch();
                for rank in &worker.ranks {
                    let target = RoutableTarget {
                        instance_id: Arc::from(worker.instance.as_str()),
                        dp_rank: *rank,
                    };
                    let incarnation = pool
                        .cursors
                        .get(&target)
                        .map_or(0, |cursor| cursor.incarnation);
                    pool.index
                        .clear(&owner(&target, incarnation), ClearScope::Worker);
                    pool.trust.remove(&target);
                    pool.cursors.remove(&target);
                }
                empty = pool.cursors.is_empty();
            }
            if empty {
                if let Some(pool) = state.pools.remove(key) {
                    state.announced = state.announced.saturating_sub(pool.announced);
                }
            }
        }
    }

    fn release_discovery_locked(state: &mut State, url: &str, generation: u64) {
        if state
            .workers
            .get(url)
            .is_some_and(|worker| worker.generation == generation)
        {
            Self::remove_locked(state, url);
        }
    }

    pub async fn discover(
        self: &Arc<Self>,
        client: &reqwest::Client,
        url: &str,
        api_key: Option<&str>,
    ) -> Result<(), String> {
        self.config.validate()?;
        let generation = {
            let mut state = self.state.lock();
            Self::remove_locked(&mut state, url);
            if state.workers.len() >= self.config.max_sources {
                return Err("KV worker capacity exhausted".into());
            }
            state.generation = state
                .generation
                .checked_add(1)
                .ok_or("KV generation exhausted")?;
            let generation = state.generation;
            state.workers.insert(
                url.to_string(),
                WorkerSource {
                    generation,
                    key: None,
                    instance: String::new(),
                    models: Vec::new(),
                    ranks: Vec::new(),
                    endpoints: Vec::new(),
                    tasks: Vec::new(),
                },
            );
            generation
        };
        let metadata = match tokio::time::timeout(
            Duration::from_secs(10),
            discover_metadata(client, url, api_key),
        )
        .await
        {
            Ok(Ok(metadata)) => metadata,
            Ok(Err(error)) => {
                let mut state = self.state.lock();
                Self::release_discovery_locked(&mut state, url, generation);
                return Err(error);
            }
            Err(_) => {
                let mut state = self.state.lock();
                Self::release_discovery_locked(&mut state, url, generation);
                return Err("KV discovery timeout".into());
            }
        };
        let (key, instance, models, sources) = metadata;
        let attach_result = {
            let mut state = self.state.lock();
            Self::attach_locked(
                self, &mut state, url, generation, key, instance, models, sources,
            )
        };
        if attach_result.is_err() {
            let mut state = self.state.lock();
            Self::release_discovery_locked(&mut state, url, generation);
        }
        attach_result
    }

    fn attach_locked(
        self: &Arc<Self>,
        state: &mut State,
        url: &str,
        generation: u64,
        key: CacheKey,
        instance: String,
        models: Vec<String>,
        sources: Vec<PublisherSource>,
    ) -> Result<(), String> {
        if state
            .workers
            .get(url)
            .is_none_or(|w| w.generation != generation)
        {
            return Err("KV discovery lease revoked".into());
        }
        let existing_sources: usize = state.pools.values().map(|p| p.cursors.len()).sum();
        if existing_sources.saturating_add(sources.len()) > self.config.max_sources {
            return Err("KV source capacity exhausted".into());
        }
        let mut seen_endpoints = HashSet::new();
        let mut seen_ranks = HashSet::new();
        for source in &sources {
            if !seen_ranks.insert(source.rank)
                || !seen_endpoints.insert(source.endpoint.as_str())
                || source
                    .replay_endpoint
                    .as_ref()
                    .is_some_and(|endpoint| !seen_endpoints.insert(endpoint.as_str()))
                || state.workers.iter().any(|(other_url, w)| {
                    other_url != url
                        && (w.instance == instance
                            || w.endpoints.contains(&source.endpoint)
                            || source
                                .replay_endpoint
                                .as_ref()
                                .is_some_and(|endpoint| w.endpoints.contains(endpoint)))
                })
            {
                return Err(
                    "KV instances, publisher and replay endpoints must be independently addressable"
                        .into(),
                );
            }
            if state.pools.get(&key).is_some_and(|p| {
                p.cursors.contains_key(&RoutableTarget {
                    instance_id: Arc::from(instance.as_str()),
                    dp_rank: source.rank,
                })
            }) {
                return Err("duplicate KV routing target".into());
            }
        }
        {
            let worker = state
                .workers
                .get_mut(url)
                .expect("discovery lease held under lock");
            worker.key = Some(key.clone());
            worker.instance = instance.clone();
            worker.models = models;
            worker.ranks = sources.iter().map(|source| source.rank).collect();
            worker.endpoints = sources
                .iter()
                .flat_map(|source| {
                    std::iter::once(source.endpoint.clone()).chain(source.replay_endpoint.clone())
                })
                .collect();
        }
        if let Some(pool) = state.pools.get_mut(&key) {
            pool.touch();
        } else {
            let id = state
                .allocate_pool_id()
                .ok_or("KV pool identity exhausted")?;
            state.pools.insert(
                key.clone(),
                Pool {
                    id,
                    revision: 0,
                    index: KvBlockIndexer::new(),
                    trust: KvTrustRegistry::new(self.config.max_sources)
                        .expect("validated capacity"),
                    cursors: HashMap::new(),
                    portable_identity: None,
                    snapshot_checksums: HashMap::new(),
                    announced: 0,
                },
            );
        }
        let pool = state.pools.get_mut(&key).expect("pool inserted above");
        for source in &sources {
            let target = RoutableTarget {
                instance_id: Arc::from(instance.as_str()),
                dp_rank: source.rank,
            };
            if !pool.trust.register(target.clone(), 0, Instant::now()) {
                return Err("KV trust capacity exhausted".into());
            }
            pool.cursors.insert(target, Cursor::new(0));
        }
        for source in sources {
            let weak = Arc::downgrade(self);
            let context = self.context.clone();
            let base = url.to_string();
            let task = tokio::spawn(subscribe(
                weak,
                context,
                base,
                generation,
                source.rank,
                source.endpoint,
                source.topic,
                source.replay_endpoint,
            ))
            .abort_handle();
            state
                .workers
                .get_mut(url)
                .expect("discovery lease held under lock")
                .tasks
                .push(task);
        }
        Ok(())
    }

    fn invalidate(&self, url: &str, generation: u64, rank: u32) {
        let mut state = self.state.lock();
        let Some(worker) = state
            .workers
            .get(url)
            .filter(|w| w.generation == generation)
        else {
            return;
        };
        let Some(key) = worker.key.clone() else {
            return;
        };
        let target = RoutableTarget {
            instance_id: Arc::from(worker.instance.as_str()),
            dp_rank: rank,
        };
        if let Some(pool) = state.pools.get_mut(&key) {
            pool.touch();
            pool.trust.mark_unavailable(&target);
            let incarnation = pool
                .cursors
                .get(&target)
                .map_or(0, |cursor| cursor.incarnation);
            pool.index
                .clear(&owner(&target, incarnation), ClearScope::Worker);
        }
    }

    fn sequence_action(
        &self,
        url: &str,
        generation: u64,
        rank: u32,
        sequence: u64,
        bytes: &[u8],
    ) -> SequenceAction {
        let state = self.state.lock();
        let Some(worker) = state
            .workers
            .get(url)
            .filter(|w| w.generation == generation && w.ranks.contains(&rank))
        else {
            return SequenceAction::Drop;
        };
        let Some(key) = worker.key.as_ref() else {
            return SequenceAction::Drop;
        };
        let target = RoutableTarget {
            instance_id: Arc::from(worker.instance.as_str()),
            dp_rank: rank,
        };
        let Some(pool) = state.pools.get(key) else {
            return SequenceAction::Drop;
        };
        let Some(cursor) = pool.cursors.get(&target) else {
            return SequenceAction::Drop;
        };
        if cursor.sequence == Some(sequence) {
            return SequenceAction::Duplicate;
        }
        // A clear is an authoritative barrier even when it fills a gap. A
        // malformed batch is left to ingestion's existing fail-closed path.
        if let Ok((batch, _)) =
            decode_runtime_batch(bytes, key, rank, self.config.max_event_age_secs, false)
        {
            if batch
                .events
                .iter()
                .any(|event| matches!(event, KVEvent::AllBlocksCleared(_)))
            {
                return SequenceAction::Apply;
            }
        }
        match cursor.sequence {
            None if sequence == 0 => SequenceAction::Apply,
            None => SequenceAction::Replay {
                from: 0,
                through: sequence - 1,
            },
            Some(last) if sequence < last && sequence == 0 => SequenceAction::Apply,
            Some(last) if sequence < last => SequenceAction::Replay {
                from: 0,
                through: sequence - 1,
            },
            Some(last) if sequence == last + 1 => SequenceAction::Apply,
            Some(last) => SequenceAction::Replay {
                from: last + 1,
                through: sequence - 1,
            },
        }
    }

    fn begin_replay(
        &self,
        url: &str,
        generation: u64,
        rank: u32,
        live_sequence: u64,
    ) -> Option<ReplayContext> {
        let mut state = self.state.lock();
        let worker = state
            .workers
            .get(url)
            .filter(|w| w.generation == generation && w.ranks.contains(&rank))?;
        let key = worker.key.clone()?;
        let target = RoutableTarget {
            instance_id: Arc::from(worker.instance.as_str()),
            dp_rank: rank,
        };
        let pool = state.pools.get_mut(&key)?;
        if !pool.cursors.contains_key(&target) {
            return None;
        }
        pool.touch();
        let cursor = pool
            .cursors
            .get_mut(&target)
            .expect("validated replay cursor");
        if cursor.sequence.is_some_and(|last| live_sequence < last) {
            let previous_incarnation = cursor.incarnation;
            cursor.reset_incarnation(cursor.incarnation.saturating_add(1));
            pool.index
                .clear(&owner(&target, previous_incarnation), ClearScope::Worker);
            pool.trust
                .register(target.clone(), cursor.incarnation, Instant::now());
        } else {
            // The existing checkpoint remains in the index, but no selection
            // may consume it while the missing suffix is being reconstructed.
            pool.trust.mark_unreconciled(&target, "replay_in_progress");
        }
        Some(ReplayContext {
            key,
            incarnation: cursor.incarnation,
        })
    }

    fn apply_replay_batch(
        &self,
        url: &str,
        generation: u64,
        rank: u32,
        sequence: u64,
        context: &ReplayContext,
        bytes: &[u8],
    ) -> Option<Instant> {
        let (batch, observed_at) = decode_runtime_batch(
            bytes,
            &context.key,
            rank,
            self.config.max_event_age_secs,
            true,
        )
        .ok()?;
        let mut state = self.state.lock();
        let worker = state
            .workers
            .get(url)
            .filter(|w| w.generation == generation && w.ranks.contains(&rank))?;
        if worker.key.as_ref() != Some(&context.key) {
            return None;
        }
        let target = RoutableTarget {
            instance_id: Arc::from(worker.instance.as_str()),
            dp_rank: rank,
        };
        let total_announced = state.announced;
        let pool = state.pools.get_mut(&context.key)?;
        let cursor = pool.cursors.get(&target)?;
        if cursor.incarnation != context.incarnation || cursor.sequence != sequence.checked_sub(1) {
            return None;
        }
        pool.touch();
        let additions = batch_additions(&batch);
        if additions > self.config.max_blocks.saturating_sub(total_announced) {
            metrics::counter!("router_kv_ingestion_failures_total", "reason" => "index_budget")
                .increment(1);
            let released = pool.announced;
            pool.index = KvBlockIndexer::new();
            pool.announced = 0;
            for target in pool.cursors.keys() {
                pool.trust.mark_unavailable(target);
            }
            state.announced = state.announced.saturating_sub(released);
            return None;
        }
        apply_decoded_batch(
            pool,
            &context.key,
            &target,
            context.incarnation,
            sequence,
            observed_at,
            batch,
            false,
        );
        let cursor = pool
            .cursors
            .get_mut(&target)
            .expect("validated replay cursor");
        cursor.sequence = Some(sequence);
        cursor.snapshot_source_sha256 = None;
        pool.announced += additions;
        state.announced += additions;
        Some(observed_at)
    }

    fn abandon_replay(&self, url: &str, generation: u64, rank: u32, context: &ReplayContext) {
        let mut state = self.state.lock();
        let Some(worker) = state
            .workers
            .get(url)
            .filter(|w| w.generation == generation && w.ranks.contains(&rank))
        else {
            return;
        };
        let target = RoutableTarget {
            instance_id: Arc::from(worker.instance.as_str()),
            dp_rank: rank,
        };
        let Some(pool) = state.pools.get_mut(&context.key) else {
            return;
        };
        if !pool.cursors.contains_key(&target) {
            return;
        }
        pool.touch();
        let cursor = pool
            .cursors
            .get_mut(&target)
            .expect("validated replay cursor");
        let previous_incarnation = cursor.incarnation;
        cursor.reset_incarnation(cursor.incarnation.saturating_add(1));
        pool.index
            .clear(&owner(&target, previous_incarnation), ClearScope::Worker);
        pool.trust
            .register(target, cursor.incarnation, Instant::now());
        metrics::counter!("router_kv_recovery_total", "result" => "failed").increment(1);
    }

    fn finish_replay(
        &self,
        url: &str,
        generation: u64,
        rank: u32,
        sequence: u64,
        context: &ReplayContext,
        observed_at: Instant,
    ) -> bool {
        let mut state = self.state.lock();
        let Some(worker) = state
            .workers
            .get(url)
            .filter(|w| w.generation == generation && w.ranks.contains(&rank))
        else {
            return false;
        };
        if worker.key.as_ref() != Some(&context.key) {
            return false;
        }
        let target = RoutableTarget {
            instance_id: Arc::from(worker.instance.as_str()),
            dp_rank: rank,
        };
        let Some(pool) = state.pools.get_mut(&context.key) else {
            return false;
        };
        if pool.cursors.get(&target).is_none_or(|cursor| {
            cursor.incarnation != context.incarnation || cursor.sequence != Some(sequence)
        }) {
            return false;
        }
        pool.touch();
        pool.trust
            .reconcile(&target, context.incarnation, sequence, observed_at)
    }

    async fn recover_and_ingest(
        &self,
        context: Arc<zmq::Context>,
        url: &str,
        generation: u64,
        rank: u32,
        sequence: u64,
        live_bytes: &[u8],
        replay_endpoint: &str,
        topic: &str,
        from: u64,
        through: u64,
    ) {
        let Some(replay_context) = self.begin_replay(url, generation, rank, sequence) else {
            return;
        };
        let dealer = (|| {
            let socket = context.socket(zmq::DEALER)?;
            socket.set_rcvhwm(256)?;
            socket.set_maxmsgsize(16 * 1024 * 1024)?;
            socket.set_linger(0)?;
            socket.connect(replay_endpoint)?;
            Ok::<_, zmq::Error>(socket)
        })();
        let Ok(dealer) = dealer else {
            self.abandon_replay(url, generation, rank, &replay_context);
            return;
        };
        if dealer
            .send_multipart([Vec::new(), from.to_be_bytes().to_vec()], zmq::DONTWAIT)
            .is_err()
        {
            self.abandon_replay(url, generation, rank, &replay_context);
            return;
        }
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        let mut expected = from;
        let mut observed_at = None;
        while expected <= through {
            if tokio::time::Instant::now() >= deadline {
                self.abandon_replay(url, generation, rank, &replay_context);
                return;
            }
            match dealer.recv_multipart(zmq::DONTWAIT) {
                Ok(frames) => {
                    if frames.len() != 4
                        || !frames[0].is_empty()
                        || frames[1] != topic.as_bytes()
                        || frames[2].len() != 8
                    {
                        self.abandon_replay(url, generation, rank, &replay_context);
                        return;
                    }
                    let replayed = u64::from_be_bytes(
                        frames[2][..].try_into().expect("checked sequence size"),
                    );
                    if replayed == u64::MAX || replayed != expected {
                        self.abandon_replay(url, generation, rank, &replay_context);
                        return;
                    }
                    let Some(seen_at) = self.apply_replay_batch(
                        url,
                        generation,
                        rank,
                        replayed,
                        &replay_context,
                        &frames[3],
                    ) else {
                        self.abandon_replay(url, generation, rank, &replay_context);
                        return;
                    };
                    observed_at = Some(seen_at);
                    expected += 1;
                    tokio::task::yield_now().await;
                }
                Err(zmq::Error::EAGAIN) => {
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
                Err(_) => {
                    self.abandon_replay(url, generation, rank, &replay_context);
                    return;
                }
            }
        }
        let Some(observed_at) = observed_at else {
            self.abandon_replay(url, generation, rank, &replay_context);
            return;
        };
        if !self.finish_replay(url, generation, rank, through, &replay_context, observed_at) {
            self.abandon_replay(url, generation, rank, &replay_context);
            return;
        }
        metrics::counter!("router_kv_recovery_total", "result" => "replayed").increment(1);
        self.ingest(url, generation, rank, sequence, live_bytes);
    }

    fn ingest(&self, url: &str, generation: u64, rank: u32, sequence: u64, bytes: &[u8]) {
        let mut state = self.state.lock();
        let Some(worker) = state
            .workers
            .get(url)
            .filter(|w| w.generation == generation && w.ranks.contains(&rank))
        else {
            return;
        };
        let Some(key) = worker.key.clone() else {
            return;
        };
        let target = RoutableTarget {
            instance_id: Arc::from(worker.instance.as_str()),
            dp_rank: rank,
        };
        let total_announced = state.announced;
        let Some(pool) = state.pools.get_mut(&key) else {
            return;
        };
        let batch = decode_runtime_batch(bytes, &key, rank, self.config.max_event_age_secs, false);
        let Ok((batch, observed_at)) = batch else {
            metrics::counter!("router_kv_ingestion_failures_total", "reason" => "invalid_or_stale_batch").increment(1);
            pool.touch();
            pool.trust.mark_unavailable(&target);
            let incarnation = pool
                .cursors
                .get(&target)
                .map_or(0, |cursor| cursor.incarnation);
            pool.index
                .clear(&owner(&target, incarnation), ClearScope::Worker);
            return;
        };
        let Some(cursor) = pool.cursors.get(&target) else {
            return;
        };
        if cursor.sequence == Some(sequence) {
            return;
        }
        pool.touch();
        let incarnation = {
            let cursor = pool
                .cursors
                .get_mut(&target)
                .expect("validated ingestion cursor");
            if cursor.sequence.is_some_and(|last| sequence < last) {
                let previous_incarnation = cursor.incarnation;
                cursor.reset_incarnation(cursor.incarnation.saturating_add(1));
                pool.index
                    .clear(&owner(&target, previous_incarnation), ClearScope::Worker);
                pool.trust
                    .register(target.clone(), cursor.incarnation, observed_at);
            }
            cursor.incarnation
        };
        let additions = batch_additions(&batch);
        if additions > self.config.max_blocks.saturating_sub(total_announced) {
            metrics::counter!("router_kv_ingestion_failures_total", "reason" => "index_budget")
                .increment(1);
            let released = pool.announced;
            pool.index = KvBlockIndexer::new();
            pool.announced = 0;
            for t in pool.cursors.keys() {
                pool.trust.mark_unavailable(t);
            }
            state.announced = state.announced.saturating_sub(released);
            return;
        }
        apply_decoded_batch(
            pool,
            &key,
            &target,
            incarnation,
            sequence,
            observed_at,
            batch,
            true,
        );
        let cursor = pool
            .cursors
            .get_mut(&target)
            .expect("validated ingestion cursor");
        cursor.sequence = Some(sequence);
        cursor.snapshot_source_sha256 = None;
        pool.announced += additions;
        state.announced += additions;
    }

    /// Selection and event publication share the same lock. No dropped signal
    /// or delayed arbiter can expose a gap's old residency as exact.
    pub fn select(
        &self,
        workers: &[Arc<dyn Worker>],
        model: Option<&str>,
        tokens: Option<&[u32]>,
        namespace_safe: bool,
    ) -> Option<(usize, KvRoutingDecision)> {
        let state = self.state.lock();
        let eligible: Vec<_> = workers
            .iter()
            .enumerate()
            .filter(|(_, w)| w.is_available())
            .filter(|(_, w)| {
                model.is_none_or(|model| {
                    (matches!(w.model_id(), "" | "unknown") || w.model_id() == model)
                        && state.workers.get(w.base_url()).is_none_or(|info| {
                            info.models.is_empty() || info.models.iter().any(|name| name == model)
                        })
                })
            })
            .collect();
        let fallback = |reason| {
            let (idx, worker) = eligible.iter().min_by(|(_, a), (_, b)| {
                a.load().cmp(&b.load()).then_with(|| a.url().cmp(b.url()))
            })?;
            Some((
                *idx,
                KvRoutingDecision {
                    target: state
                        .workers
                        .get(worker.base_url())
                        .filter(|i| !i.instance.is_empty())
                        .map(|info| RoutableTarget {
                            instance_id: Arc::from(info.instance.as_str()),
                            dp_rank: worker.dp_rank().unwrap_or(0) as u32,
                        }),
                    mode: KvRoutingMode::Fallback,
                    fallback_reason: Some(reason),
                    scores: Vec::new(),
                },
            ))
        };
        if self.config.validate().is_err() {
            return fallback("invalid_configuration");
        }
        if !namespace_safe {
            return fallback("unsupported_request_namespace");
        }
        let Some(tokens) = tokens else {
            return fallback("missing_prepared_tokens");
        };
        let mut key = None;
        let mut candidates = Vec::new();
        for (_, worker) in &eligible {
            let Some(info) = state.workers.get(worker.base_url()) else {
                return fallback("undiscovered_worker");
            };
            let Some(cache_key) = &info.key else {
                return fallback("unavailable_metadata");
            };
            if key.as_ref().is_some_and(|k| k != cache_key) {
                return fallback("cache_identity_mismatch");
            }
            key = Some(cache_key.clone());
            let rank = worker.dp_rank().unwrap_or(0) as u32;
            if !info.ranks.contains(&rank) {
                return fallback("unaddressable_rank");
            }
            candidates.push(KvRoutingCandidate {
                target: RoutableTarget {
                    instance_id: Arc::from(info.instance.as_str()),
                    dp_rank: rank,
                },
                active_requests: worker.load() as u64,
                healthy: true,
            });
        }
        let key = key?;
        let pool = state.pools.get(&key)?;
        // Do not credit partial blocks or the final token required for logits.
        let full =
            tokens.len().saturating_sub(1) / key.block_size as usize * key.block_size as usize;
        let query = MatchQuery {
            group_idx: 0,
            local_hashes: local_hashes(&tokens[..full], key.block_size, None)
                .iter()
                .map(|h| hex_key(h))
                .collect(),
            tiers_of_interest: vec![StorageTier::Device],
        };
        let selector = TrustedKvSelector::new(
            self.config.cache_weight,
            1.0,
            Duration::from_secs(self.config.max_event_age_secs),
        )
        .ok()?;
        let decision = selector.select(
            &pool.index,
            &query,
            &candidates,
            &pool.trust,
            Instant::now(),
        );
        let selected = decision.target.as_ref()?;
        let index = candidates.iter().position(|c| &c.target == selected)?;
        Some((eligible[index].0, decision))
    }
}

fn owner(target: &RoutableTarget, incarnation: u64) -> ResidencyOwner {
    ResidencyOwner::Worker {
        source: target.instance_id.clone(),
        dp_rank: target.dp_rank,
        incarnation,
    }
}

fn unix_millis() -> Result<u64, PortableSnapshotError> {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| PortableSnapshotError::Invalid("invalid system clock"))?
        .as_millis();
    u64::try_from(millis)
        .map_err(|_| PortableSnapshotError::Invalid("system clock exceeds snapshot range"))
}

fn snapshot_observed_instant(
    observed_at_unix_ms: u64,
    now_unix_ms: u64,
    now: Instant,
    max_event_age_secs: u64,
) -> Result<Instant, PortableSnapshotError> {
    let age_ms = now_unix_ms.saturating_sub(observed_at_unix_ms);
    // Preserve staleness rather than laundering an old snapshot into fresh
    // credit. One extra second guarantees projection as stale at the boundary.
    let cap_ms = max_event_age_secs
        .saturating_mul(1_000)
        .saturating_add(1_000);
    now.checked_sub(Duration::from_millis(age_ms.min(cap_ms)))
        .ok_or(PortableSnapshotError::Invalid(
            "snapshot observation exceeds the monotonic clock range",
        ))
}

fn stage_snapshot_source(
    index: &KvBlockIndexer,
    source: &ValidatedPortableSource,
    now_unix_ms: u64,
    now: Instant,
    max_event_age_secs: u64,
) -> Result<(), PortableSnapshotError> {
    // The validator proved that the parent graph is acyclic. Re-walk it here in
    // dependency order so `store` never skips a suffix with an unknown parent.
    let _ = snapshot_observed_instant(
        source.observed_at_unix_ms,
        now_unix_ms,
        now,
        max_event_age_secs,
    )?;
    let mut pending: Vec<_> = source
        .residencies
        .iter()
        .flat_map(|residency| {
            residency
                .prefixes
                .iter()
                .map(move |prefix| (residency, prefix))
        })
        .collect();
    let mut known = HashSet::<Arc<str>>::new();
    while !pending.is_empty() {
        let before = pending.len();
        let mut next = Vec::new();
        for (residency, prefix) in pending {
            if prefix
                .parent_block_hash
                .as_ref()
                .is_some_and(|parent| !known.contains(parent))
            {
                next.push((residency, prefix));
                continue;
            }
            index.store(
                0,
                residency.owner.clone(),
                residency.tier,
                prefix.parent_block_hash.clone(),
                &prefix.blocks,
            );
            known.extend(prefix.blocks.iter().map(|(hash, _)| hash.clone()));
        }
        if next.len() == before {
            return Err(PortableSnapshotError::Invalid(
                "validated snapshot prefix graph made no progress",
            ));
        }
        pending = next;
    }
    Ok(())
}

fn hex_key(bytes: &[u8]) -> Arc<str> {
    Arc::from(bytes.iter().map(|b| format!("{b:02x}")).collect::<String>())
}

fn decode_runtime_batch(
    bytes: &[u8],
    key: &CacheKey,
    rank: u32,
    max_event_age_secs: u64,
    allow_stale: bool,
) -> Result<(KVEventBatch, Instant), String> {
    let batch = decode_batch(bytes)?;
    validate_device_batch(&batch, key.block_size as usize)?;
    if batch.data_parallel_rank.is_some_and(|r| r != rank) {
        return Err("publisher rank mismatch".into());
    }
    let observed_at = if allow_stale {
        publisher_replay_observed_at(batch.ts, max_event_age_secs)?
    } else {
        publisher_observed_at(batch.ts, max_event_age_secs)?
    };
    Ok((batch, observed_at))
}

fn batch_additions(batch: &KVEventBatch) -> usize {
    batch
        .events
        .iter()
        .map(|event| match event {
            KVEvent::BlockStored(stored) => stored.block_hashes.len(),
            _ => 0,
        })
        .sum()
}

fn apply_decoded_batch(
    pool: &mut Pool,
    key: &CacheKey,
    target: &RoutableTarget,
    incarnation: u64,
    sequence: u64,
    observed_at: Instant,
    batch: KVEventBatch,
    reconcile_after_clear: bool,
) {
    pool.trust
        .advance(target, incarnation, sequence, observed_at);
    let residency = owner(target, incarnation);
    for event in batch.events {
        match event {
            KVEvent::AllBlocksCleared(clear) => {
                if let Some(medium) = clear.medium.as_deref() {
                    let tier = StorageTier::parse_medium(medium)
                        .expect("device admission rejects unknown clear mediums");
                    pool.index.clear_tier(&residency, ClearScope::Worker, tier);
                } else {
                    pool.index.clear(&residency, ClearScope::Worker);
                }
                if reconcile_after_clear {
                    pool.trust
                        .reconcile(target, incarnation, sequence, observed_at);
                }
            }
            KVEvent::BlockStored(stored) => {
                let locals = local_hashes(&stored.token_ids, key.block_size, None);
                let blocks: Vec<_> = stored
                    .block_hashes
                    .iter()
                    .zip(locals)
                    .map(|(hash, local)| (opaque(hash), hex_key(&local)))
                    .collect();
                pool.index.store(
                    0,
                    residency.clone(),
                    StorageTier::Device,
                    stored.parent_block_hash.as_ref().map(opaque),
                    &blocks,
                );
            }
            KVEvent::BlockRemoved(removed) => pool.index.remove(
                0,
                &residency,
                StorageTier::Device,
                &removed.block_hashes.iter().map(opaque).collect::<Vec<_>>(),
            ),
        }
    }
}

fn publisher_observed_at(timestamp: f64, max_event_age_secs: u64) -> Result<Instant, String> {
    let wall = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| "invalid clock")?
        .as_secs_f64();
    if timestamp > wall + 1.0 || wall - timestamp > max_event_age_secs as f64 {
        return Err("stale publisher timestamp".into());
    }
    let age = Duration::try_from_secs_f64((wall - timestamp).max(0.0))
        .map_err(|_| "publisher timestamp outside monotonic-clock range")?;
    Instant::now()
        .checked_sub(age)
        .ok_or_else(|| "publisher event age outside monotonic-clock range".into())
}

fn publisher_replay_observed_at(
    timestamp: f64,
    max_event_age_secs: u64,
) -> Result<Instant, String> {
    let wall = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| "invalid clock")?
        .as_secs_f64();
    if !timestamp.is_finite() || timestamp > wall + 1.0 {
        return Err("invalid replay timestamp".into());
    }
    let age = (wall - timestamp).max(0.0);
    // Replaying an old but internally contiguous history is valid. Preserve
    // its age so the reconciled source immediately projects as stale instead
    // of laundering old observations into fresh exact credit.
    let age = if age > max_event_age_secs as f64 {
        Duration::from_secs(max_event_age_secs.saturating_add(1))
    } else {
        Duration::try_from_secs_f64(age)
            .map_err(|_| "publisher timestamp outside monotonic-clock range")?
    };
    Instant::now()
        .checked_sub(age)
        .ok_or_else(|| "publisher event age outside monotonic-clock range".into())
}

fn opaque(hash: &super::wire::ExternalBlockHash) -> Arc<str> {
    match hash {
        super::wire::ExternalBlockHash::Bytes(bytes) => Arc::from(format!("b:{}", hex_key(bytes))),
        super::wire::ExternalBlockHash::Int(i) => Arc::from(format!("i:{i}")),
    }
}

type Discovery = (CacheKey, String, Vec<String>, Vec<PublisherSource>);

async fn discover_metadata(
    client: &reqwest::Client,
    worker_url: &str,
    api_key: Option<&str>,
) -> Result<Discovery, String> {
    let (info, sources, models) = if crate::backend::is_grpc_url(worker_url) {
        let timeout = Duration::from_secs(8);
        let server = crate::backend::control::get_grpc_server_info(worker_url, timeout).await?;
        let model = crate::backend::control::get_grpc_model_info(worker_url, timeout).await?;
        let source =
            crate::backend::control::get_grpc_kv_event_sources(worker_url, timeout).await?;
        let info = serde_json::json!({ "instance_id":server.instance_id, "model_id":model.model_id, "kv_block_size":server.kv_block_size });
        let mut sources = KvEventSourcesResponse::new();
        for source in source.sources {
            if source.transport != "zmq" {
                continue;
            }
            let rank = source.data_parallel_rank.ok_or("missing publisher rank")?;
            if sources
                .insert(
                    rank.to_string(),
                    super::KvEventSourceEntry {
                        endpoint: source.endpoint,
                        replay_endpoint: (!source.replay_endpoint.is_empty())
                            .then_some(source.replay_endpoint),
                        topic: source.topic,
                        publisher: Some("zmq".into()),
                        ..Default::default()
                    },
                )
                .is_some()
            {
                return Err("duplicate publisher rank".into());
            }
        }
        let mut models = model.served_model_aliases;
        models.push(model.served_model_name);
        (info, sources, models)
    } else {
        let mut request = client.get(format!("{worker_url}/get_server_info"));
        if let Some(key) = api_key {
            request = request.bearer_auth(key);
        }
        let info: Value = request
            .send()
            .await
            .map_err(|e| e.to_string())?
            .error_for_status()
            .map_err(|e| e.to_string())?
            .json()
            .await
            .map_err(|e| e.to_string())?;
        let mut request = client.get(format!("{worker_url}/kv_event_sources"));
        if let Some(key) = api_key {
            request = request.bearer_auth(key);
        }
        let sources: KvEventSourcesResponse = request
            .send()
            .await
            .map_err(|e| e.to_string())?
            .error_for_status()
            .map_err(|e| e.to_string())?
            .json()
            .await
            .map_err(|e| e.to_string())?;
        (info, sources, Vec::new())
    };
    let instance = info["instance_id"]
        .as_str()
        .filter(|s| !s.is_empty() && s.len() <= 256)
        .ok_or("missing worker incarnation identity")?
        .to_string();
    let model = info["model_id"]
        .as_str()
        .or_else(|| info["model_path"].as_str())
        .filter(|s| !s.is_empty() && s.len() <= 4096)
        .ok_or("missing worker model identity")?;
    let block_size = info["kv_block_size"]
        .as_u64()
        .or_else(|| info["effective_attention_block_size"].as_u64())
        .or_else(|| info["block_size"].as_u64())
        .and_then(|s| u32::try_from(s).ok())
        .filter(|s| *s > 0)
        .ok_or("missing valid KV block size")?;
    let mut models = models;
    models.push(model.to_string());
    models.retain(|m| !m.is_empty());
    let http_origin = worker_url.replacen("grpc://", "http://", 1);
    let parsed = url::Url::parse(&http_origin).map_err(|e| e.to_string())?;
    let host = parsed.host().ok_or("missing worker host")?.to_string();
    let mut publishers = Vec::new();
    for (rank, entry) in sources {
        if entry.publisher.as_deref().is_some_and(|p| p != "zmq")
            || entry.enable_kv_cache_events == Some(false)
            || entry.endpoint.is_empty()
        {
            continue;
        }
        let rank = rank.parse::<u32>().map_err(|_| "invalid publisher rank")?;
        let endpoint = entry
            .endpoint
            .replace("tcp://*:", &format!("tcp://{host}:"))
            .replace("tcp://0.0.0.0:", &format!("tcp://{host}:"));
        let (_, endpoint) = super::parse_endpoint_mapping(&format!("{http_origin}={endpoint}"))?;
        let replay_endpoint = entry
            .replay_endpoint
            .filter(|replay| !replay.is_empty())
            .map(|replay| {
                let replay = replay
                    .replace("tcp://*:", &format!("tcp://{host}:"))
                    .replace("tcp://0.0.0.0:", &format!("tcp://{host}:"));
                super::parse_endpoint_mapping(&format!("{http_origin}={replay}"))
                    .map(|(_, endpoint)| endpoint)
            })
            .transpose()?;
        if entry.topic.len() > 1024 {
            return Err("publisher topic too long".into());
        }
        publishers.push(PublisherSource {
            rank,
            endpoint,
            topic: entry.topic,
            replay_endpoint,
        });
    }
    if publishers.is_empty() {
        return Err("no supported KV event publishers".into());
    }
    Ok((
        CacheKey {
            model: Arc::from(model),
            hash_mode: HashMode::Sha256,
            block_size,
        },
        instance,
        models,
        publishers,
    ))
}

async fn subscribe(
    runtime: Weak<KvRuntime>,
    context: Arc<zmq::Context>,
    url: String,
    generation: u64,
    rank: u32,
    endpoint: String,
    topic: String,
    replay_endpoint: Option<String>,
) {
    let socket = (|| {
        let socket = context.socket(zmq::SUB)?;
        socket.set_rcvhwm(256)?;
        socket.set_maxmsgsize(16 * 1024 * 1024)?;
        socket.set_linger(0)?;
        socket.set_subscribe(topic.as_bytes())?;
        socket.connect(&endpoint)?;
        Ok::<_, zmq::Error>(socket)
    })();
    let Ok(socket) = socket else {
        if let Some(runtime) = runtime.upgrade() {
            runtime.invalidate(&url, generation, rank);
        }
        return;
    };
    loop {
        match socket.recv_multipart(zmq::DONTWAIT) {
            Ok(frames) => {
                let Some(runtime) = runtime.upgrade() else {
                    return;
                };
                if frames.len() != 3 || frames[0] != topic.as_bytes() || frames[1].len() != 8 {
                    runtime.invalidate(&url, generation, rank);
                } else {
                    let sequence = u64::from_be_bytes(
                        frames[1][..].try_into().expect("checked sequence size"),
                    );
                    if sequence > i64::MAX as u64 {
                        runtime.invalidate(&url, generation, rank);
                    } else {
                        match runtime.sequence_action(&url, generation, rank, sequence, &frames[2])
                        {
                            SequenceAction::Apply => {
                                runtime.ingest(&url, generation, rank, sequence, &frames[2]);
                            }
                            SequenceAction::Duplicate | SequenceAction::Drop => {}
                            SequenceAction::Replay { from, through } => {
                                if let Some(replay_endpoint) = replay_endpoint.as_deref() {
                                    runtime
                                        .recover_and_ingest(
                                            context.clone(),
                                            &url,
                                            generation,
                                            rank,
                                            sequence,
                                            &frames[2],
                                            replay_endpoint,
                                            &topic,
                                            from,
                                            through,
                                        )
                                        .await;
                                } else {
                                    runtime.ingest(&url, generation, rank, sequence, &frames[2]);
                                }
                            }
                        }
                    }
                }
                tokio::task::yield_now().await;
            }
            Err(zmq::Error::EAGAIN) => tokio::time::sleep(Duration::from_millis(5)).await,
            Err(_) => {
                if let Some(runtime) = runtime.upgrade() {
                    runtime.invalidate(&url, generation, rank);
                }
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replay_of_old_history_reconciles_as_stale_not_fresh() {
        let wall = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        let observed = publisher_replay_observed_at(wall - 30.0, 1).unwrap();
        assert!(observed.elapsed() >= Duration::from_secs(2));
        assert!(publisher_replay_observed_at(wall + 10.0, 1).is_err());
    }

    #[test]
    fn portable_snapshot_respects_the_global_block_budget() {
        let runtime = KvRuntime::new(crate::config::types::TrustedKvAwareConfig {
            max_blocks: 4,
            ..Default::default()
        });
        let expected_identity = PortableCacheIdentity::new(
            "m",
            "tokenizer-v1",
            super::super::snapshot::PortableHashMode::Sha256,
            2,
        );
        let matching_key = CacheKey {
            model: Arc::from("m"),
            hash_mode: HashMode::Sha256,
            block_size: 2,
        };
        let other_key = CacheKey {
            model: Arc::from("other"),
            hash_mode: HashMode::Sha256,
            block_size: 2,
        };
        let target = RoutableTarget {
            instance_id: Arc::from("worker-0"),
            dp_rank: 0,
        };
        {
            let mut state = runtime.state.lock();
            state.announced = 3;
            state.pools.insert(
                other_key,
                Pool {
                    id: 1,
                    revision: 0,
                    index: KvBlockIndexer::new(),
                    trust: KvTrustRegistry::new(8).unwrap(),
                    cursors: HashMap::new(),
                    portable_identity: None,
                    snapshot_checksums: HashMap::new(),
                    announced: 3,
                },
            );
            state.pools.insert(
                matching_key,
                Pool {
                    id: 2,
                    revision: 0,
                    index: KvBlockIndexer::new(),
                    trust: KvTrustRegistry::new(8).unwrap(),
                    cursors: HashMap::from([(target.clone(), Cursor::new(0))]),
                    portable_identity: None,
                    snapshot_checksums: HashMap::new(),
                    announced: 0,
                },
            );
        }
        let now = unix_millis().unwrap();
        let mut snapshot = PortableKvSnapshot {
            format_version: super::super::snapshot::PORTABLE_SNAPSHOT_FORMAT_VERSION,
            snapshot_id: "snapshot-budget".into(),
            complete: true,
            cache_identity: expected_identity.clone(),
            created_at_unix_ms: now,
            fence: PortableSnapshotFence::new("router-a", 1),
            sources: vec![super::super::snapshot::PortableSourceSnapshot {
                instance_id: "worker-0".into(),
                dp_rank: 0,
                incarnation: 1,
                sequence: 1,
                observed_at_unix_ms: now,
                residencies: vec![super::super::snapshot::PortableResidency {
                    owner: super::super::snapshot::PortableResidencyOwner::Worker {
                        instance_id: "worker-0".into(),
                        dp_rank: 0,
                        incarnation: 1,
                    },
                    tier: super::super::snapshot::PortableStorageTier::Device,
                    prefixes: vec![super::super::snapshot::PortablePrefix {
                        parent_block_hash: None,
                        blocks: vec![
                            super::super::snapshot::PortableBlock {
                                block_hash: "b:01".into(),
                                local_hash: "l:01".into(),
                            },
                            super::super::snapshot::PortableBlock {
                                block_hash: "b:02".into(),
                                local_hash: "l:02".into(),
                            },
                        ],
                    }],
                }],
            }],
            checksum_sha256: String::new(),
        };
        snapshot.seal().unwrap();
        assert_eq!(
            runtime
                .import_portable_snapshot(
                    &snapshot,
                    &expected_identity,
                    &PortableSnapshotFence::new("router-a", 1),
                )
                .unwrap_err(),
            PortableSnapshotError::BlockBudgetExceeded
        );
        let state = runtime.state.lock();
        assert_eq!(state.announced, 3);
        assert_eq!(
            state
                .pools
                .values()
                .map(|pool| pool.announced)
                .sum::<usize>(),
            3
        );
    }
}
