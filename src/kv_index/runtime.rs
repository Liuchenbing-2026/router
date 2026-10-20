//! Fenced ingestion and selection for normal-dense, worker-owned device KV.
//!
//! A validated AllBlocksCleared is the completeness barrier. Live deltas alone
//! never establish completeness. Request metadata cannot publish residency.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;
use serde_json::Value;

use super::routing::{
    KvRoutingCandidate, KvRoutingDecision, KvRoutingMode, KvTrustRegistry, TrustedKvSelector,
};
use super::{
    decode_batch, local_hashes, validate_device_batch, CacheKey, ClearScope, HashMode, KVEvent,
    KvBlockIndexer, KvEventSourcesResponse, MatchQuery, ResidencyOwner, RoutableTarget,
    StorageTier,
};
use crate::config::types::TrustedKvAwareConfig;
use crate::core::Worker;

struct Cursor {
    sequence: Option<u64>,
    incarnation: u64,
}

struct Pool {
    index: KvBlockIndexer,
    trust: KvTrustRegistry,
    cursors: HashMap<RoutableTarget, Cursor>,
    // A conservative allocation budget: removals do not refund historical
    // insertions because alias/topology metadata may outlive residency.
    announced: usize,
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
    workers: HashMap<String, WorkerSource>,
    pools: HashMap<CacheKey, Pool>,
    announced: usize,
}

pub struct KvRuntime {
    config: TrustedKvAwareConfig,
    state: Mutex<State>,
    context: Arc<zmq::Context>,
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
        Self {
            config,
            state: Mutex::new(State::default()),
            context: Arc::new(zmq::Context::new()),
        }
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
                for rank in &worker.ranks {
                    let target = RoutableTarget {
                        instance_id: Arc::from(worker.instance.as_str()),
                        dp_rank: *rank,
                    };
                    pool.index.clear(&owner(&target, 0), ClearScope::Worker);
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
        let metadata = tokio::time::timeout(
            Duration::from_secs(10),
            discover_metadata(client, url, api_key),
        )
        .await
        .map_err(|_| "KV discovery timeout".to_string())??;
        let (key, instance, models, sources) = metadata;
        let mut state = self.state.lock();
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
        let mut seen = HashSet::new();
        let mut seen_ranks = HashSet::new();
        for (rank, endpoint, _) in &sources {
            if !seen_ranks.insert(*rank)
                || !seen.insert(endpoint.as_str())
                || state.workers.iter().any(|(other_url, w)| {
                    other_url != url && (w.instance == instance || w.endpoints.contains(endpoint))
                })
            {
                return Err(
                    "KV instances and publisher endpoints must be independently addressable".into(),
                );
            }
            if state.pools.get(&key).is_some_and(|p| {
                p.cursors.contains_key(&RoutableTarget {
                    instance_id: Arc::from(instance.as_str()),
                    dp_rank: *rank,
                })
            }) {
                return Err("duplicate KV routing target".into());
            }
        }
        let pool = state.pools.entry(key.clone()).or_insert_with(|| Pool {
            index: KvBlockIndexer::new(),
            trust: KvTrustRegistry::new(self.config.max_sources).expect("validated capacity"),
            cursors: HashMap::new(),
            announced: 0,
        });
        for (rank, _, _) in &sources {
            let target = RoutableTarget {
                instance_id: Arc::from(instance.as_str()),
                dp_rank: *rank,
            };
            if !pool.trust.register(target.clone(), 0, Instant::now()) {
                return Err("KV trust capacity exhausted".into());
            }
            pool.cursors.insert(
                target,
                Cursor {
                    sequence: None,
                    incarnation: 0,
                },
            );
        }
        let worker = state
            .workers
            .get_mut(url)
            .expect("discovery lease held under lock");
        worker.key = Some(key);
        worker.instance = instance;
        worker.models = models;
        for (rank, endpoint, topic) in sources {
            worker.ranks.push(rank);
            worker.endpoints.push(endpoint.clone());
            let weak = Arc::downgrade(self);
            let context = self.context.clone();
            let base = url.to_string();
            worker.tasks.push(
                tokio::spawn(subscribe(
                    weak, context, base, generation, rank, endpoint, topic,
                ))
                .abort_handle(),
            );
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
            pool.trust.mark_unavailable(&target);
            pool.index.clear(&owner(&target, 0), ClearScope::Worker);
        }
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
        let batch = decode_batch(bytes).and_then(|batch| {
            validate_device_batch(&batch, key.block_size as usize)?;
            if batch.data_parallel_rank.is_some_and(|r| r != rank) {
                return Err("publisher rank mismatch".into());
            }
            let wall = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|_| "invalid clock")?
                .as_secs_f64();
            if batch.ts > wall + 1.0 || wall - batch.ts > self.config.max_event_age_secs as f64 {
                return Err("stale publisher timestamp".into());
            }
            Ok(batch)
        });
        let Ok(batch) = batch else {
            metrics::counter!("router_kv_ingestion_failures_total", "reason" => "invalid_or_stale_batch").increment(1);
            pool.trust.mark_unavailable(&target);
            pool.index.clear(&owner(&target, 0), ClearScope::Worker);
            return;
        };
        let Some(cursor) = pool.cursors.get_mut(&target) else {
            return;
        };
        if cursor.sequence == Some(sequence) {
            return;
        }
        if cursor.sequence.is_some_and(|last| sequence < last) {
            cursor.incarnation = cursor.incarnation.saturating_add(1);
            cursor.sequence = None;
            pool.index.clear(&owner(&target, 0), ClearScope::Worker);
            pool.trust
                .register(target.clone(), cursor.incarnation, Instant::now());
        }
        let now = Instant::now();
        pool.trust
            .advance(&target, cursor.incarnation, sequence, now);
        let residency = owner(&target, cursor.incarnation);
        let mut reconciled = pool.trust.is_reconciled(&target, cursor.incarnation);
        if !reconciled {
            // A gap also revokes previously published claims. Selection's
            // fallback alone cannot isolate topology in the shared index.
            pool.index.clear(&residency, ClearScope::Worker);
        }
        // Ignore pre-barrier stores for both index publication and its budget:
        // an incomplete source must not evict a reconciled peer's index.
        // Preserve event order when clear and stores share a batch.
        let additions: usize = batch
            .events
            .iter()
            .map(|event| match event {
                KVEvent::AllBlocksCleared(_) => {
                    reconciled = true;
                    0
                }
                KVEvent::BlockStored(s) if reconciled => s.block_hashes.len(),
                _ => 0,
            })
            .sum();
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
        for event in batch.events {
            match event {
                KVEvent::AllBlocksCleared(_) => {
                    pool.index.clear(&residency, ClearScope::Worker);
                    pool.trust
                        .reconcile(&target, cursor.incarnation, sequence, now);
                }
                KVEvent::BlockStored(s)
                    if pool.trust.is_reconciled(&target, cursor.incarnation) =>
                {
                    let locals = local_hashes(&s.token_ids, key.block_size, None);
                    let blocks: Vec<_> = s
                        .block_hashes
                        .iter()
                        .zip(locals)
                        .map(|(seq, local)| (opaque(seq), hex_key(&local)))
                        .collect();
                    pool.index.store(
                        0,
                        residency.clone(),
                        StorageTier::Device,
                        s.parent_block_hash.as_ref().map(opaque),
                        &blocks,
                    );
                }
                KVEvent::BlockRemoved(r)
                    if pool.trust.is_reconciled(&target, cursor.incarnation) =>
                {
                    pool.index.remove(
                        0,
                        &residency,
                        StorageTier::Device,
                        &r.block_hashes.iter().map(opaque).collect::<Vec<_>>(),
                    )
                }
                // Do not create or rewrite shared aliases before a validated
                // completeness barrier, including after loss or restart.
                KVEvent::BlockStored(_) | KVEvent::BlockRemoved(_) => {}
            }
        }
        cursor.sequence = Some(sequence);
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

fn hex_key(bytes: &[u8]) -> Arc<str> {
    Arc::from(bytes.iter().map(|b| format!("{b:02x}")).collect::<String>())
}

fn opaque(hash: &super::wire::ExternalBlockHash) -> Arc<str> {
    match hash {
        super::wire::ExternalBlockHash::Bytes(bytes) => Arc::from(format!("b:{}", hex_key(bytes))),
        super::wire::ExternalBlockHash::Int(i) => Arc::from(format!("i:{i}")),
    }
}

type Discovery = (CacheKey, String, Vec<String>, Vec<(u32, String, String)>);

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
        if entry.topic.len() > 1024 {
            return Err("publisher topic too long".into());
        }
        publishers.push((rank, endpoint, entry.topic));
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
                        runtime.ingest(&url, generation, rank, sequence, &frames[2]);
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
mod human_review_tests {
    use super::*;
    use crate::core::BasicWorker;
    use serde_json::json;

    fn cache_key() -> CacheKey {
        CacheKey {
            model: Arc::from("m"),
            hash_mode: HashMode::Sha256,
            block_size: 2,
        }
    }

    fn fixture(max_blocks: usize) -> KvRuntime {
        let runtime = KvRuntime::new(TrustedKvAwareConfig {
            max_blocks,
            ..Default::default()
        });
        let mut pool = Pool {
            index: KvBlockIndexer::new(),
            trust: KvTrustRegistry::new(2).unwrap(),
            cursors: HashMap::new(),
            announced: 0,
        };
        let mut state = runtime.state.lock();
        for (name, generation) in [("a", 1), ("b", 2)] {
            let target = RoutableTarget {
                instance_id: Arc::from(name),
                dp_rank: 0,
            };
            assert!(pool.trust.register(target.clone(), 0, Instant::now()));
            pool.cursors.insert(
                target,
                Cursor {
                    sequence: None,
                    incarnation: 0,
                },
            );
            state.workers.insert(
                format!("http://{name}"),
                WorkerSource {
                    generation,
                    key: Some(cache_key()),
                    instance: name.to_string(),
                    models: vec!["m".into()],
                    ranks: vec![0],
                    endpoints: Vec::new(),
                    tasks: Vec::new(),
                },
            );
        }
        state.pools.insert(cache_key(), pool);
        drop(state);
        runtime
    }

    fn stored(hashes: &[u64], parent: Option<u64>, tokens: &[u32]) -> KVEvent {
        let KVEvent::BlockStored(mut event) = serde_json::from_value(json!({"type":"BlockStored", "block_hashes":hashes, "parent_block_hash":parent, "token_ids":tokens, "block_size":2, "lora_id":null, "medium":"GPU", "lora_name":null})).unwrap() else { unreachable!() };
        // Production admission only accepts 32-byte sha256_cbor hash keys.
        let hash = |value: u64| {
            super::super::wire::ExternalBlockHash::Bytes(vec![u8::try_from(value).unwrap(); 32])
        };
        event.block_hashes = hashes.iter().map(|value| hash(*value)).collect();
        event.parent_block_hash = parent.map(hash);
        KVEvent::BlockStored(event)
    }

    fn clear() -> KVEvent {
        KVEvent::AllBlocksCleared(super::super::AllBlocksCleared::default())
    }

    fn send(runtime: &KvRuntime, name: &str, sequence: u64, events: Vec<KVEvent>) {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        let bytes = rmp_serde::to_vec_named(&(ts, events, 0_u32)).unwrap();
        validate_device_batch(&decode_batch(&bytes).unwrap(), 2)
            .expect("regression fixture must satisfy production admission");
        runtime.ingest(
            &format!("http://{name}"),
            if name == "a" { 1 } else { 2 },
            0,
            sequence,
            &bytes,
        );
    }

    fn held(runtime: &KvRuntime, name: &str, tokens: &[u32]) -> u32 {
        let query = MatchQuery {
            group_idx: 0,
            local_hashes: local_hashes(tokens, 2, None)
                .iter()
                .map(|h| hex_key(h))
                .collect(),
            tiers_of_interest: vec![StorageTier::Device],
        };
        runtime.state.lock().pools[&cache_key()]
            .index
            .find_matches(&query)
            .iter()
            .filter(|hit| hit.target.instance_id.as_ref() == name)
            .flat_map(|hit| &hit.segments)
            .map(|(start, end)| end - start)
            .sum()
    }

    fn announced(runtime: &KvRuntime) -> usize {
        let state = runtime.state.lock();
        assert_eq!(state.announced, state.pools[&cache_key()].announced);
        state.announced
    }

    fn decision(runtime: &KvRuntime, names: &[&str], tokens: &[u32]) -> KvRoutingDecision {
        let workers: Vec<Arc<dyn Worker>> = names
            .iter()
            .map(|name| {
                Arc::new(BasicWorker::new(
                    format!("http://{name}"),
                    crate::core::WorkerType::Regular,
                )) as Arc<dyn Worker>
            })
            .collect();
        runtime
            .select(&workers, Some("m"), Some(tokens), true)
            .unwrap()
            .1
    }

    #[test]
    fn unknown_parent_then_other_source_chain_and_clear_leaves_no_stale_credit() {
        // The literal review sequence: A attaches mid-cache with Y's parent X
        // missing, B publishes X -> Y, then A clears. Main's parent guard must
        // not leave A's suffix Y at the root or disturb B's legitimate chain.
        let runtime = fixture(16);
        send(&runtime, "a", 1, vec![stored(&[20], Some(10), &[3, 4])]);
        send(
            &runtime,
            "b",
            1,
            vec![clear(), stored(&[10, 20], None, &[1, 2, 3, 4])],
        );
        send(&runtime, "a", 2, vec![clear()]);
        assert_eq!(held(&runtime, "a", &[3, 4]), 0);
        assert_eq!(held(&runtime, "b", &[1, 2, 3, 4]), 2);
        let d = decision(&runtime, &["a", "b"], &[3, 4, 5]);
        assert_eq!(d.mode, KvRoutingMode::Exact);
        assert!(d.scores.iter().all(|s| s.credited_blocks == 0.0));
    }

    #[test]
    fn unreconciled_root_deltas_never_enter_the_shared_index() {
        let runtime = fixture(16);
        send(&runtime, "a", 1, vec![stored(&[10], None, &[1, 2])]);
        assert_eq!(
            held(&runtime, "a", &[1, 2]),
            0,
            "unreconciled source published residency"
        );
        assert_eq!(announced(&runtime), 0);
        send(
            &runtime,
            "b",
            1,
            vec![clear(), stored(&[10, 20], None, &[1, 2, 3, 4])],
        );
        send(&runtime, "a", 2, vec![stored(&[10], None, &[1, 2])]);
        assert_eq!(held(&runtime, "a", &[1, 2]), 0);
        assert_eq!(held(&runtime, "b", &[1, 2, 3, 4]), 2);
        assert_eq!(announced(&runtime), 2);
        assert_eq!(
            decision(&runtime, &["a", "b"], &[1, 2, 3, 4, 5]).mode,
            KvRoutingMode::Fallback
        );
        send(&runtime, "a", 3, vec![clear()]);
        let d = decision(&runtime, &["a", "b"], &[1, 2, 3, 4, 5]);
        assert_eq!(d.mode, KvRoutingMode::Exact);
        assert_eq!(d.target.unwrap().instance_id.as_ref(), "b");
    }

    #[test]
    fn unreconciled_source_cannot_exhaust_a_healthy_sources_shared_budget() {
        let runtime = fixture(2);
        send(
            &runtime,
            "b",
            1,
            vec![clear(), stored(&[10, 20], None, &[1, 2, 3, 4])],
        );
        assert_eq!(held(&runtime, "b", &[1, 2, 3, 4]), 2);
        send(&runtime, "a", 1, vec![stored(&[30], None, &[7, 8])]);
        assert_eq!(
            held(&runtime, "b", &[1, 2, 3, 4]),
            2,
            "unreconciled delta wiped another source's verified cache"
        );
        assert_eq!(announced(&runtime), 2);
        assert_eq!(
            decision(&runtime, &["b"], &[1, 2, 3, 4, 5]).mode,
            KvRoutingMode::Exact
        );
    }

    #[test]
    fn a_sequence_gap_discards_old_claims_and_quarantines_later_deltas() {
        let runtime = fixture(16);
        send(
            &runtime,
            "a",
            1,
            vec![clear(), stored(&[10], None, &[1, 2])],
        );
        send(
            &runtime,
            "b",
            1,
            vec![clear(), stored(&[10, 20], None, &[1, 2, 3, 4])],
        );
        send(&runtime, "a", 3, vec![stored(&[30], None, &[7, 8])]);
        assert_eq!(
            held(&runtime, "a", &[1, 2]),
            0,
            "gap retained pre-gap claims in the shared index"
        );
        assert_eq!(held(&runtime, "a", &[7, 8]), 0);
        assert_eq!(held(&runtime, "b", &[1, 2, 3, 4]), 2);
        send(&runtime, "a", 4, vec![stored(&[30], None, &[7, 8])]);
        assert_eq!(announced(&runtime), 3);
        send(
            &runtime,
            "a",
            5,
            vec![clear(), stored(&[30], None, &[7, 8])],
        );
        assert_eq!(held(&runtime, "a", &[7, 8]), 1);
        assert_eq!(announced(&runtime), 4);
        assert_eq!(
            decision(&runtime, &["a", "b"], &[1, 2, 3, 4, 5]).mode,
            KvRoutingMode::Exact
        );
    }

    #[test]
    fn only_stores_after_a_clear_in_the_same_batch_are_published_and_counted() {
        let runtime = fixture(1);
        send(
            &runtime,
            "a",
            1,
            vec![
                stored(&[10], None, &[1, 2]),
                clear(),
                stored(&[30], None, &[7, 8]),
            ],
        );
        assert_eq!(held(&runtime, "a", &[1, 2]), 0);
        assert_eq!(
            held(&runtime, "a", &[7, 8]),
            1,
            "pre-barrier delta consumed the only admitted block slot"
        );
        assert_eq!(announced(&runtime), 1);
        assert_eq!(
            decision(&runtime, &["a"], &[7, 8, 9]).mode,
            KvRoutingMode::Exact
        );
    }

    #[test]
    fn invalidation_and_rollback_require_a_new_clear_before_publication() {
        let runtime = fixture(16);
        send(
            &runtime,
            "a",
            1,
            vec![clear(), stored(&[10], None, &[1, 2])],
        );
        send(
            &runtime,
            "b",
            1,
            vec![clear(), stored(&[10, 20], None, &[1, 2, 3, 4])],
        );
        runtime.invalidate("http://a", 1, 0);
        send(&runtime, "a", 2, vec![stored(&[30], None, &[7, 8])]);
        assert_eq!(
            held(&runtime, "a", &[7, 8]),
            0,
            "unavailable source republished without a clear"
        );
        send(
            &runtime,
            "a",
            3,
            vec![clear(), stored(&[30], None, &[7, 8])],
        );
        assert_eq!(held(&runtime, "a", &[7, 8]), 1);
        send(&runtime, "a", 1, vec![stored(&[10], None, &[1, 2])]);
        assert_eq!(
            held(&runtime, "a", &[1, 2]),
            0,
            "new incarnation republished without a clear"
        );
        assert_eq!(held(&runtime, "a", &[7, 8]), 0);
        assert_eq!(held(&runtime, "b", &[1, 2, 3, 4]), 2);
        send(
            &runtime,
            "a",
            2,
            vec![clear(), stored(&[40], None, &[9, 10])],
        );
        assert_eq!(held(&runtime, "a", &[9, 10]), 1);
        assert_eq!(announced(&runtime), 5);
    }
}
