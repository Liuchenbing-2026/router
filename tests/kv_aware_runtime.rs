//! Real TCP ZMQ ingestion, HTTP forwarding, lifecycle and configuration checks.
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json,
};
use serde_json::{json, Value};
use vllm_router_rs::{
    config::{ConfigValidator, PolicyConfig, RouterConfig, RoutingMode, TrustedKvAwareConfig},
    core::{BasicWorker, Worker, WorkerType},
    kv_index::{
        local_hashes, routing::KvRoutingMode, wire::ExternalBlockHash, AllBlocksCleared,
        BlockRemoved, BlockStored, KVEvent, PortableBlock, PortableCacheIdentity, PortableHashMode,
        PortableKvSnapshot, PortablePrefix, PortableResidency, PortableResidencyOwner,
        PortableSnapshotError, PortableSnapshotFence, PortableSnapshotFenceAuthority,
        PortableSnapshotImportStatus, PortableSourceSnapshot, PortableStorageTier,
        PORTABLE_SNAPSHOT_FORMAT_VERSION,
    },
    policies::{PolicyFactory, TrustedKvAwarePolicy},
    protocols::spec::CompletionRequest,
    routers::http::router::Router,
    server::AppContext,
};

struct Publisher {
    _context: zmq::Context,
    socket: zmq::Socket,
    replay_socket: Option<zmq::Socket>,
    endpoint: String,
    replay_endpoint: Option<String>,
}

#[derive(Debug, Default)]
struct SharedFenceAuthority {
    current: Mutex<Option<PortableSnapshotFence>>,
}

impl PortableSnapshotFenceAuthority for SharedFenceAuthority {
    fn admit_and_advance(
        &self,
        fence: &PortableSnapshotFence,
    ) -> Result<(), PortableSnapshotError> {
        let mut current = self.current.lock().unwrap();
        if let Some(applied) = current.as_ref() {
            if fence.epoch < applied.epoch {
                return Err(PortableSnapshotError::StaleEpoch);
            }
            if fence.epoch == applied.epoch && fence.owner_id != applied.owner_id {
                return Err(PortableSnapshotError::EpochOwnerConflict);
            }
        }
        *current = Some(fence.clone());
        Ok(())
    }
}

#[derive(Debug)]
struct BlockingFenceAuthority {
    current: Mutex<Option<PortableSnapshotFence>>,
    entered: mpsc::Sender<()>,
    release: Mutex<Option<mpsc::Receiver<()>>>,
}

impl PortableSnapshotFenceAuthority for BlockingFenceAuthority {
    fn admit_and_advance(
        &self,
        fence: &PortableSnapshotFence,
    ) -> Result<(), PortableSnapshotError> {
        let mut current = self.current.lock().unwrap();
        if let Some(applied) = current.as_ref() {
            if fence.epoch < applied.epoch {
                return Err(PortableSnapshotError::StaleEpoch);
            }
            if fence.epoch == applied.epoch && fence.owner_id != applied.owner_id {
                return Err(PortableSnapshotError::EpochOwnerConflict);
            }
        }
        *current = Some(fence.clone());
        drop(current);

        let release = self.release.lock().unwrap().take();
        if let Some(release) = release {
            let _ = self.entered.send(());
            let _ = release.recv();
        }
        Ok(())
    }
}

impl Publisher {
    fn new() -> Self {
        Self::make(false)
    }
    fn with_replay() -> Self {
        Self::make(true)
    }
    fn make(with_replay: bool) -> Self {
        let context = zmq::Context::new();
        let socket = context.socket(zmq::XPUB).unwrap();
        socket.set_linger(0).unwrap();
        socket.bind("tcp://127.0.0.1:*").unwrap();
        let endpoint = socket.get_last_endpoint().unwrap().unwrap();
        let (replay_socket, replay_endpoint) = if with_replay {
            let replay = context.socket(zmq::ROUTER).unwrap();
            replay.set_linger(0).unwrap();
            replay.bind("tcp://127.0.0.1:*").unwrap();
            let endpoint = replay.get_last_endpoint().unwrap().unwrap();
            (Some(replay), Some(endpoint))
        } else {
            (None, None)
        };
        Self {
            _context: context,
            socket,
            replay_socket,
            endpoint,
            replay_endpoint,
        }
    }
    async fn subscribed(&self) {
        let until = Instant::now() + Duration::from_secs(3);
        loop {
            if let Ok(message) = self.socket.recv_bytes(zmq::DONTWAIT) {
                if message == b"\x01kv" {
                    return;
                }
            }
            assert!(Instant::now() < until, "subscriber did not attach");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }
    fn send(&self, rank: u32, sequence: u64, events: Vec<KVEvent>) {
        self.send_at(
            rank,
            sequence,
            events,
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
        );
    }
    fn send_at(&self, rank: u32, sequence: u64, events: Vec<KVEvent>, timestamp: f64) {
        let bytes = batch_at(rank, events, timestamp);
        self.raw(sequence, &bytes);
    }
    fn raw(&self, sequence: u64, bytes: &[u8]) {
        self.socket
            .send_multipart([b"kv".as_slice(), &sequence.to_be_bytes(), bytes], 0)
            .unwrap();
    }
    async fn replay_request(&self, expected_from: u64) -> Vec<Vec<u8>> {
        let socket = self
            .replay_socket
            .as_ref()
            .expect("publisher replay socket");
        let deadline = Instant::now() + Duration::from_secs(3);
        let request = loop {
            match socket.recv_multipart(zmq::DONTWAIT) {
                Ok(request) => break request,
                Err(zmq::Error::EAGAIN) => {
                    assert!(Instant::now() < deadline, "replay request did not arrive");
                    tokio::time::sleep(Duration::from_millis(2)).await;
                }
                Err(error) => panic!("replay request failed: {error}"),
            }
        };
        assert_eq!(request.len(), 3);
        assert_eq!(
            u64::from_be_bytes(request[2][..].try_into().unwrap()),
            expected_from
        );
        request
    }
    fn send_replay(&self, request: &[Vec<u8>], replies: &[(u64, Vec<KVEvent>)]) {
        self.send_replay_with_delimiter(request, replies, &[]);
    }
    fn send_replay_with_delimiter(
        &self,
        request: &[Vec<u8>],
        replies: &[(u64, Vec<KVEvent>)],
        delimiter: &[u8],
    ) {
        let socket = self
            .replay_socket
            .as_ref()
            .expect("publisher replay socket");
        for (sequence, events) in replies {
            let payload = batch_at(
                0,
                events.clone(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64(),
            );
            socket
                .send_multipart(
                    [
                        request[0].clone(),
                        delimiter.to_vec(),
                        b"kv".to_vec(),
                        sequence.to_be_bytes().to_vec(),
                        payload,
                    ],
                    0,
                )
                .unwrap();
        }
    }
    async fn serve_replay(&self, expected_from: u64, replies: &[(u64, Vec<KVEvent>)]) {
        let request = self.replay_request(expected_from).await;
        self.send_replay(&request, replies);
    }
}
fn batch_at(rank: u32, events: Vec<KVEvent>, timestamp: f64) -> Vec<u8> {
    rmp_serde::to_vec_named(&(timestamp, events, rank)).unwrap()
}
fn clear() -> KVEvent {
    KVEvent::AllBlocksCleared(AllBlocksCleared::default())
}
fn stored() -> KVEvent {
    KVEvent::BlockStored(BlockStored {
        extra_keys: None,
        block_hashes: vec![
            ExternalBlockHash::Bytes(vec![1; 32]),
            ExternalBlockHash::Bytes(vec![2; 32]),
        ],
        parent_block_hash: None,
        token_ids: vec![1, 2, 3, 4],
        block_size: 2,
        lora_id: None,
        medium: Some("GPU".into()),
        lora_name: None,
        group_idx: None,
        kv_cache_spec_kind: None,
        kv_cache_spec_sliding_window: None,
        locality: None,
        ownership: None,
        session_id: None,
    })
}
fn removed() -> KVEvent {
    KVEvent::BlockRemoved(BlockRemoved {
        block_hashes: vec![ExternalBlockHash::Bytes(vec![1; 32])],
        medium: Some("GPU".into()),
        group_idx: None,
        locality: None,
        ownership: None,
    })
}

#[derive(Clone)]
struct MockState {
    info: Value,
    sources: Value,
    captured: Arc<Mutex<Vec<(HeaderMap, Value)>>>,
}
struct Engine {
    url: String,
    state: MockState,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for Engine {
    fn drop(&mut self) {
        self.task.abort();
    }
}
impl Engine {
    async fn new(instance: &str, publishers: &[(u32, &Publisher)]) -> Self {
        Self::with_cache(instance, publishers, "m", 2).await
    }

    async fn with_cache(
        instance: &str,
        publishers: &[(u32, &Publisher)],
        model: &str,
        block_size: u32,
    ) -> Self {
        let sources: serde_json::Map<String, Value> = publishers
            .iter()
            .map(|(rank, publisher)| {
                (
                    rank.to_string(),
                    json!({
                        "endpoint": publisher.endpoint,
                        "replay_endpoint": publisher.replay_endpoint,
                        "topic": "kv",
                        "publisher": "zmq",
                        "enable_kv_cache_events": true
                    }),
                )
            })
            .collect();
        let state = MockState {
            info: json!({
                "instance_id": instance,
                "model_id": model,
                "kv_block_size": block_size
            }),
            sources: Value::Object(sources),
            captured: Arc::new(Mutex::new(Vec::new())),
        };
        let app = axum::Router::new()
            .route("/health", get(|| async { "ok" }))
            .route(
                "/get_server_info",
                get(|State(state): State<MockState>| async move { Json(state.info) }),
            )
            .route(
                "/kv_event_sources",
                get(|State(state): State<MockState>| async move { Json(state.sources) }),
            )
            .route("/v1/completions", post(complete))
            .with_state(state.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        Self { url, state, task }
    }
    fn worker(&self) -> Arc<dyn Worker> {
        Arc::new(BasicWorker::new(self.url.clone(), WorkerType::Regular))
    }
}
async fn complete(
    State(state): State<MockState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    state
        .captured
        .lock()
        .unwrap()
        .push((headers.clone(), body.clone()));
    if body["prompt"][0] == 999 {
        return std::future::pending().await;
    }
    if body["prompt"][0] == 500 {
        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
    }
    if body["stream"] == true {
        let pending = futures_util::stream::pending::<Result<bytes::Bytes, std::io::Error>>();
        return (
            [("content-type", "text/event-stream")],
            Body::from_stream(pending),
        )
            .into_response();
    }
    Json(json!({"worker":state.info["instance_id"],"rank":headers.get("x-data-parallel-rank").and_then(|h|h.to_str().ok()),"prompt":body["prompt"]})).into_response()
}
fn policy() -> TrustedKvAwarePolicy {
    TrustedKvAwarePolicy::with_config(TrustedKvAwareConfig {
        cache_weight: 0.9,
        ..Default::default()
    })
}
async fn wait_for(
    policy: &TrustedKvAwarePolicy,
    workers: &[Arc<dyn Worker>],
    expected: KvRoutingMode,
) -> (usize, vllm_router_rs::kv_index::routing::KvRoutingDecision) {
    let until = Instant::now() + Duration::from_secs(3);
    loop {
        let result = policy
            .select_with_tokens(workers, Some("m"), Some(&[1, 2, 3, 4, 5]), true)
            .unwrap();
        if result.1.mode == expected {
            return result;
        }
        assert!(
            Instant::now() < until,
            "mode {:?}, expected {expected:?}",
            result.1
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}
async fn attach(policy: &TrustedKvAwarePolicy, engine: &Engine, publishers: &[&Publisher]) {
    policy
        .runtime
        .discover(&reqwest::Client::new(), &engine.url, None)
        .await
        .unwrap();
    for publisher in publishers {
        publisher.subscribed().await;
    }
}
async fn router(engines: &[&Engine], dp_size: usize) -> (Arc<Router>, Arc<AppContext>) {
    let urls: Vec<_> = engines.iter().map(|e| e.url.clone()).collect();
    let config = RouterConfig {
        mode: RoutingMode::Regular {
            worker_urls: urls.clone(),
        },
        policy: PolicyConfig::TrustedKvAware {
            config: TrustedKvAwareConfig {
                cache_weight: 0.9,
                ..Default::default()
            },
        },
        intra_node_data_parallel_size: dp_size,
        worker_startup_timeout_secs: 3,
        worker_startup_check_interval_secs: 1,
        disable_retries: true,
        ..Default::default()
    };
    let ctx =
        Arc::new(AppContext::new(config, reqwest::Client::new(), 64, None, Vec::new()).unwrap());
    (Arc::new(Router::new(urls, &ctx).await.unwrap()), ctx)
}
fn request(prompt: Value, stream: bool) -> CompletionRequest {
    serde_json::from_value(json!({"model":"m","prompt":prompt,"max_tokens":1,"stream":stream}))
        .unwrap()
}

#[tokio::test]
async fn real_event_ingestion_requires_clear_then_prefers_holding_worker_and_honors_eviction() {
    let a = Publisher::new();
    let b = Publisher::new();
    let e0 = Engine::new("idle", &[(0, &a)]).await;
    let e1 = Engine::new("holder", &[(0, &b)]).await;
    let policy = policy();
    let workers = vec![e0.worker(), e1.worker()];
    attach(&policy, &e0, &[&a]).await;
    attach(&policy, &e1, &[&b]).await;
    b.send(0, 1, vec![stored()]);
    assert_eq!(
        wait_for(&policy, &workers, KvRoutingMode::Fallback)
            .await
            .1
            .mode,
        KvRoutingMode::Fallback
    );
    a.send(0, 1, vec![clear()]);
    assert_eq!(
        wait_for(&policy, &workers, KvRoutingMode::Degraded)
            .await
            .1
            .mode,
        KvRoutingMode::Degraded
    );
    b.send(0, 2, vec![clear(), stored()]);
    assert_eq!(wait_for(&policy, &workers, KvRoutingMode::Exact).await.0, 1);
    // Advertised model aliases cannot broaden an explicit registry model pool.
    let other_label: Arc<dyn Worker> = Arc::new(
        BasicWorker::new(e1.url.clone(), WorkerType::Regular).with_labels(
            std::collections::HashMap::from([("model_id".into(), "other".into())]),
        ),
    );
    assert!(policy
        .select_with_tokens(&[other_label], Some("m"), Some(&[1, 2, 3, 4, 5]), true)
        .is_none());
    b.send(0, 3, vec![removed()]);
    let until = Instant::now() + Duration::from_secs(3);
    loop {
        let d = wait_for(&policy, &workers, KvRoutingMode::Exact).await.1;
        if d.scores.iter().all(|s| s.credited_blocks == 0.0) {
            break;
        }
        assert!(Instant::now() < until);
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

#[tokio::test]
async fn gaps_corruption_wrong_rank_and_old_timestamps_need_a_new_clear_barrier() {
    let publisher = Publisher::new();
    let engine = Engine::new("w", &[(0, &publisher)]).await;
    let policy = policy();
    let workers = vec![engine.worker()];
    attach(&policy, &engine, &[&publisher]).await;
    publisher.send(0, 1, vec![clear(), stored()]);
    wait_for(&policy, &workers, KvRoutingMode::Exact).await;
    publisher.send(0, 3, vec![stored()]);
    wait_for(&policy, &workers, KvRoutingMode::Fallback).await;
    publisher.send(0, 4, vec![stored()]);
    publisher.send(0, 5, vec![clear()]);
    wait_for(&policy, &workers, KvRoutingMode::Exact).await;
    publisher.raw(6, b"corrupt");
    wait_for(&policy, &workers, KvRoutingMode::Fallback).await;
    publisher.send(0, 7, vec![clear()]);
    wait_for(&policy, &workers, KvRoutingMode::Exact).await;
    publisher.send(1, 8, vec![stored()]);
    wait_for(&policy, &workers, KvRoutingMode::Fallback).await;
    publisher.send(0, 9, vec![clear()]);
    wait_for(&policy, &workers, KvRoutingMode::Exact).await;
    publisher.send_at(0, 10, vec![stored()], 1.0);
    wait_for(&policy, &workers, KvRoutingMode::Fallback).await;
    publisher.send(0, 1, vec![stored()]);
    publisher.send(0, 2, vec![clear()]);
    wait_for(&policy, &workers, KvRoutingMode::Exact).await;
}

#[tokio::test]
async fn late_attach_replays_history_before_trusting_live_events() {
    let publisher = Publisher::with_replay();
    let engine = Engine::new("w", &[(0, &publisher)]).await;
    let policy = policy();
    let workers = vec![engine.worker()];
    attach(&policy, &engine, &[&publisher]).await;

    // The first live sequence is one, so the runtime must reconstruct seq 0
    // before it may consume seq 1 or award exact-match credit.
    publisher.send(0, 1, vec![stored()]);
    let request = publisher.replay_request(0).await;
    assert_eq!(
        wait_for(&policy, &workers, KvRoutingMode::Fallback)
            .await
            .1
            .mode,
        KvRoutingMode::Fallback,
        "exact credit escaped while replay was incomplete"
    );
    publisher.send_replay(&request, &[(0, vec![clear()])]);

    let decision = wait_for(&policy, &workers, KvRoutingMode::Exact).await.1;
    assert_eq!(decision.scores[0].credited_blocks, 2.0);
}

#[tokio::test]
async fn trusted_runtime_replays_a_live_gap_and_restores_exact() {
    let publisher = Publisher::with_replay();
    let engine = Engine::new("w", &[(0, &publisher)]).await;
    let policy = policy();
    let workers = vec![engine.worker()];
    attach(&policy, &engine, &[&publisher]).await;

    publisher.send(0, 0, vec![clear()]);
    wait_for(&policy, &workers, KvRoutingMode::Exact).await;

    // Seq 1 is absent from live delivery. Replaying just the missing suffix
    // must preserve the seq-0 checkpoint and then apply live seq 2.
    publisher.send(0, 2, vec![stored()]);
    publisher.serve_replay(1, &[(1, vec![clear()])]).await;

    let decision = wait_for(&policy, &workers, KvRoutingMode::Exact).await.1;
    assert_eq!(decision.scores[0].credited_blocks, 2.0);
}

#[tokio::test]
async fn partial_replay_never_escapes_and_stays_fenced_until_a_clear() {
    let publisher = Publisher::with_replay();
    let engine = Engine::new("w", &[(0, &publisher)]).await;
    let policy = policy();
    let workers = vec![engine.worker()];
    attach(&policy, &engine, &[&publisher]).await;

    publisher.send(0, 3, vec![stored()]);
    // Seq 0 is applied into the fenced index, then seq 2 violates the missing
    // seq 1. None of that partial residency may become exact credit.
    publisher
        .serve_replay(
            0,
            &[(0, vec![clear(), stored()]), (2, vec![clear(), stored()])],
        )
        .await;
    assert_eq!(
        wait_for(&policy, &workers, KvRoutingMode::Fallback)
            .await
            .1
            .mode,
        KvRoutingMode::Fallback
    );

    publisher.send(0, 4, vec![clear(), stored()]);
    assert_eq!(wait_for(&policy, &workers, KvRoutingMode::Exact).await.0, 0);
}

#[tokio::test]
async fn malformed_replay_envelope_fences_then_a_clear_recovers() {
    let publisher = Publisher::with_replay();
    let engine = Engine::new("w", &[(0, &publisher)]).await;
    let policy = policy();
    let workers = vec![engine.worker()];
    attach(&policy, &engine, &[&publisher]).await;

    publisher.send(0, 1, vec![stored()]);
    let request = publisher.replay_request(0).await;
    publisher.send_replay_with_delimiter(&request, &[(0, vec![clear()])], b"not-empty");
    assert_eq!(
        wait_for(&policy, &workers, KvRoutingMode::Fallback)
            .await
            .1
            .mode,
        KvRoutingMode::Fallback
    );

    publisher.send(0, 2, vec![clear(), stored()]);
    let decision = wait_for(&policy, &workers, KvRoutingMode::Exact).await.1;
    assert_eq!(decision.scores[0].credited_blocks, 2.0);
}

#[tokio::test]
async fn rediscovery_clears_residency_from_every_source_incarnation() {
    let pa = Publisher::new();
    let pb = Publisher::new();
    let a = Engine::new("a", &[(0, &pa)]).await;
    let b = Engine::new("b", &[(0, &pb)]).await;
    let policy = policy();
    let workers = vec![a.worker(), b.worker()];
    attach(&policy, &a, &[&pa]).await;
    attach(&policy, &b, &[&pb]).await;

    pa.send(0, 1, vec![clear(), stored()]);
    pb.send(0, 1, vec![clear()]);
    wait_for(&policy, &workers, KvRoutingMode::Exact).await;

    // A sequence rollback starts incarnation 1 and leaves residency under that
    // owner. Removing the target must not clear only incarnation 0.
    pa.send(0, 0, vec![stored()]);
    wait_for(&policy, &workers, KvRoutingMode::Degraded).await;
    policy.runtime.remove_worker(&a.url);

    attach(&policy, &a, &[&pa]).await;
    pa.send(0, 1, vec![clear()]);
    let decision = wait_for(&policy, &workers, KvRoutingMode::Exact).await.1;
    let rediscovered = decision
        .scores
        .iter()
        .find(|score| score.target.instance_id.as_ref() == "a")
        .unwrap();
    assert_eq!(rediscovered.credited_blocks, 0.0);
}

#[tokio::test]
async fn freshness_duplicates_budget_and_removal_do_not_preserve_exact_credit() {
    let publisher = Publisher::new();
    let engine = Engine::new("w", &[(0, &publisher)]).await;
    let policy = TrustedKvAwarePolicy::with_config(TrustedKvAwareConfig {
        max_event_age_secs: 1,
        max_blocks: 2,
        ..Default::default()
    });
    let workers = vec![engine.worker()];
    attach(&policy, &engine, &[&publisher]).await;
    publisher.send(0, 1, vec![clear(), stored()]);
    wait_for(&policy, &workers, KvRoutingMode::Exact).await;
    tokio::time::sleep(Duration::from_millis(1100)).await;
    publisher.send(0, 1, vec![clear(), stored()]);
    wait_for(&policy, &workers, KvRoutingMode::Fallback).await;
    publisher.send(0, 2, vec![clear()]);
    wait_for(&policy, &workers, KvRoutingMode::Exact).await;
    publisher.send(0, 3, vec![stored()]);
    wait_for(&policy, &workers, KvRoutingMode::Fallback).await;
    publisher.send(0, 4, vec![clear()]);
    wait_for(&policy, &workers, KvRoutingMode::Exact).await;
    policy.runtime.remove_worker(&engine.url);
    let decision = policy
        .select_with_tokens(&workers, Some("m"), Some(&[1, 2, 3, 4, 5]), true)
        .unwrap()
        .1;
    assert_eq!(decision.fallback_reason, Some("undiscovered_worker"));
}

#[tokio::test]
async fn publisher_timestamp_age_consumes_the_receivers_freshness_window() {
    let publisher = Publisher::new();
    let engine = Engine::new("aged", &[(0, &publisher)]).await;
    let policy = TrustedKvAwarePolicy::with_config(TrustedKvAwareConfig {
        max_event_age_secs: 1,
        ..Default::default()
    });
    let workers = vec![engine.worker()];
    attach(&policy, &engine, &[&publisher]).await;
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
        - 0.6;
    publisher.send_at(0, 1, vec![clear(), stored()], timestamp);
    wait_for(&policy, &workers, KvRoutingMode::Exact).await;
    tokio::time::sleep(Duration::from_millis(650)).await;
    wait_for(&policy, &workers, KvRoutingMode::Fallback).await;
}

#[tokio::test]
async fn real_http_forwarding_selects_dp_rank_and_replaces_client_rank_header() {
    let p0 = Publisher::new();
    let p1 = Publisher::new();
    let engine = Engine::new("dp", &[(0, &p0), (1, &p1)]).await;
    let (router, ctx) = router(&[&engine], 2).await;
    p0.subscribed().await;
    p1.subscribed().await;
    let policy = ctx.policy_registry.get_default_policy();
    let kv = policy
        .as_any()
        .downcast_ref::<TrustedKvAwarePolicy>()
        .unwrap();
    let workers = ctx.worker_registry.get_all();
    p0.send(0, 1, vec![clear()]);
    p1.send(1, 1, vec![clear(), stored()]);
    wait_for(kv, &workers, KvRoutingMode::Exact).await;
    let mut headers = HeaderMap::new();
    headers.insert("x-data-parallel-rank", "0".parse().unwrap());
    let response = router
        .route_typed_request(
            Some(&headers),
            &request(json!([1, 2, 3, 4, 5]), false),
            "/v1/completions",
            Some("m"),
        )
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["x-router-kv-mode"], "exact");
    let body = axum::body::to_bytes(response.into_body(), 4096)
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(body["rank"], "1");
    assert_eq!(body["prompt"], json!([1, 2, 3, 4, 5]));
    assert!(workers.iter().all(|w| w.load() == 0));

    p0.raw(2, b"corrupt");
    wait_for(kv, &workers, KvRoutingMode::Degraded).await;
    let response = router
        .route_typed_request(
            None,
            &request(json!([1, 2, 3, 4, 5]), false),
            "/v1/completions",
            Some("m"),
        )
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["x-router-kv-mode"], "degraded");
    let body = axum::body::to_bytes(response.into_body(), 4096)
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(body["rank"], "1");
    assert!(workers.iter().all(|w| w.load() == 0));
}

#[tokio::test]
async fn http_text_and_salted_prompts_fall_back_without_claiming_exact() {
    let publisher = Publisher::new();
    let engine = Engine::new("w", &[(0, &publisher)]).await;
    let (router, ctx) = router(&[&engine], 1).await;
    publisher.subscribed().await;
    let policy = ctx.policy_registry.get_default_policy();
    let kv = policy
        .as_any()
        .downcast_ref::<TrustedKvAwarePolicy>()
        .unwrap();
    publisher.send(0, 1, vec![clear(), stored()]);
    wait_for(kv, &ctx.worker_registry.get_all(), KvRoutingMode::Exact).await;
    let response = router
        .route_typed_request(
            None,
            &request(json!("hello"), false),
            "/v1/completions",
            Some("m"),
        )
        .await;
    assert_eq!(
        response.headers()["x-router-kv-reason"],
        "missing_prepared_tokens"
    );
    let salted: CompletionRequest =
        serde_json::from_value(json!({"model":"m","prompt":[1,2,3,4,5],"cache_salt":"tenant-b"}))
            .unwrap();
    let response = router
        .route_typed_request(None, &salted, "/v1/completions", Some("m"))
        .await;
    assert_eq!(
        response.headers()["x-router-kv-reason"],
        "unsupported_request_namespace"
    );
}

#[tokio::test]
async fn stream_drop_and_cancellation_before_headers_release_exactly_one_load_slot() {
    let engine = Engine::new("w", &[]).await;
    let (router, ctx) = router(&[&engine], 1).await;
    let worker = ctx.worker_registry.get_all()[0].clone();
    let response = router
        .route_typed_request(
            None,
            &request(json!([1, 2, 3]), true),
            "/v1/completions",
            Some("m"),
        )
        .await;
    assert_eq!(worker.load(), 1);
    drop(response);
    let until = Instant::now() + Duration::from_secs(2);
    while worker.load() != 0 {
        assert!(Instant::now() < until);
        tokio::task::yield_now().await;
    }
    let pending_router = router.clone();
    let task = tokio::spawn(async move {
        pending_router
            .route_typed_request(
                None,
                &request(json!([999, 2, 3]), false),
                "/v1/completions",
                Some("m"),
            )
            .await
    });
    let until = Instant::now() + Duration::from_secs(2);
    while engine.state.captured.lock().unwrap().len() < 2 {
        assert!(Instant::now() < until);
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert_eq!(worker.load(), 1);
    task.abort();
    let _ = task.await;
    assert_eq!(worker.load(), 0);
    let response = router
        .route_typed_request(
            None,
            &request(json!([500, 2, 3]), false),
            "/v1/completions",
            Some("m"),
        )
        .await;
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(worker.load(), 0);
}

#[test]
fn config_is_registered_and_invalid_limits_and_disaggregated_modes_are_rejected() {
    let policy: PolicyConfig = serde_json::from_value(json!({"type":"trusted_kv_aware"})).unwrap();
    assert_eq!(
        PolicyFactory::create_from_config(&policy).name(),
        "trusted_kv_aware"
    );
    assert!(PolicyFactory::create_by_name("trusted_kv_aware").is_some());
    for config in [
        TrustedKvAwareConfig {
            cache_weight: f64::NAN,
            ..Default::default()
        },
        TrustedKvAwareConfig {
            max_sources: 0,
            ..Default::default()
        },
        TrustedKvAwareConfig {
            max_event_age_secs: 0,
            ..Default::default()
        },
        TrustedKvAwareConfig {
            max_blocks: 0,
            ..Default::default()
        },
    ] {
        let config = RouterConfig {
            policy: PolicyConfig::TrustedKvAware { config },
            mode: RoutingMode::Regular {
                worker_urls: vec!["http://worker:8000".into()],
            },
            ..Default::default()
        };
        assert!(ConfigValidator::validate(&config).is_err());
    }
    let config = RouterConfig {
        policy,
        enable_igw: true,
        ..Default::default()
    };
    assert!(ConfigValidator::validate(&config).is_err());
}

#[test]
fn discovery_failure_does_not_erase_registry_model_boundaries() {
    let worker = |url: &str, model: &str| -> Arc<dyn Worker> {
        Arc::new(
            BasicWorker::new(url.into(), WorkerType::Regular).with_labels(
                std::collections::HashMap::from([("model_id".into(), model.into())]),
            ),
        )
    };
    let workers = vec![
        worker("http://a:8000", "m"),
        worker("http://b:8000", "other"),
    ];
    workers[0].increment_load();
    let policy = policy();
    assert_eq!(
        policy
            .select_with_tokens(&workers, Some("m"), None, true)
            .unwrap()
            .0,
        0
    );
    assert!(policy
        .select_with_tokens(&workers, Some("unknown-model"), None, true)
        .is_none());
}

#[tokio::test]
async fn removed_worker_cannot_be_reattached_by_an_old_discovery_response() {
    let publisher = Publisher::new();
    let ready = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let (ready_handler, release_handler) = (ready.clone(), release.clone());
    let endpoint = publisher.endpoint.clone();
    let app =
        axum::Router::new()
            .route(
                "/get_server_info",
                get(move || {
                    let (ready, release) = (ready_handler.clone(), release_handler.clone());
                    async move {
                        ready.notify_one();
                        release.notified().await;
                        Json(json!({"instance_id":"delayed","model_id":"m","kv_block_size":2}))
                    }
                }),
            )
            .route(
                "/kv_event_sources",
                get(move || {
                    let endpoint = endpoint.clone();
                    async move {
                        Json(json!({"0":{"endpoint":endpoint,"topic":"kv","publisher":"zmq"}}))
                    }
                }),
            );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let policy = policy();
    let runtime = policy.runtime.clone();
    let pending_url = url.clone();
    let discovery = tokio::spawn(async move {
        runtime
            .discover(&reqwest::Client::new(), &pending_url, None)
            .await
    });
    tokio::time::timeout(Duration::from_secs(2), ready.notified())
        .await
        .unwrap();
    policy.runtime.remove_worker(&url);
    release.notify_one();
    assert!(discovery
        .await
        .unwrap()
        .unwrap_err()
        .contains("lease revoked"));
    let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(BasicWorker::new(url, WorkerType::Regular))];
    assert_eq!(
        policy
            .select_with_tokens(&workers, Some("m"), Some(&[1, 2, 3]), true)
            .unwrap()
            .1
            .fallback_reason,
        Some("undiscovered_worker")
    );
    server.abort();
}

#[tokio::test]
async fn dynamic_add_and_remove_drive_subscription_lifecycle() {
    let p0 = Publisher::new();
    let e0 = Engine::new("first", &[(0, &p0)]).await;
    let (router, ctx) = router(&[&e0], 1).await;
    p0.subscribed().await;
    let p1 = Publisher::new();
    let e1 = Engine::new("added", &[(0, &p1)]).await;
    router.add_worker(&e1.url).await.unwrap();
    p1.subscribed().await;
    p0.send(0, 1, vec![clear()]);
    p1.send(0, 1, vec![clear(), stored()]);
    let policy = ctx.policy_registry.get_default_policy();
    let kv = policy
        .as_any()
        .downcast_ref::<TrustedKvAwarePolicy>()
        .unwrap();
    wait_for(kv, &ctx.worker_registry.get_all(), KvRoutingMode::Exact).await;
    let response = router
        .route_typed_request(
            None,
            &request(json!([1, 2, 3, 4, 5]), false),
            "/v1/completions",
            Some("m"),
        )
        .await;
    let body = axum::body::to_bytes(response.into_body(), 4096)
        .await
        .unwrap();
    assert_eq!(
        serde_json::from_slice::<Value>(&body).unwrap()["worker"],
        "added"
    );
    router.remove_worker(&e1.url);
    let response = router
        .route_typed_request(
            None,
            &request(json!([1, 2, 3, 4, 5]), false),
            "/v1/completions",
            Some("m"),
        )
        .await;
    let body = axum::body::to_bytes(response.into_body(), 4096)
        .await
        .unwrap();
    assert_eq!(
        serde_json::from_slice::<Value>(&body).unwrap()["worker"],
        "first"
    );
    assert_eq!(ctx.worker_registry.get_all().len(), 1);
}

#[tokio::test]
async fn duplicate_publisher_mapping_and_source_capacity_are_rejected_without_credit() {
    let publisher = Publisher::new();
    let duplicate = Engine::new("duplicate", &[(0, &publisher), (1, &publisher)]).await;
    let policy = policy();
    assert!(policy
        .runtime
        .discover(&reqwest::Client::new(), &duplicate.url, None)
        .await
        .is_err());
    let decision = policy
        .select_with_tokens(
            &[duplicate.worker()],
            Some("m"),
            Some(&[1, 2, 3, 4, 5]),
            true,
        )
        .unwrap()
        .1;
    assert_eq!(decision.mode, KvRoutingMode::Fallback);
    let p0 = Publisher::new();
    let p1 = Publisher::new();
    let two = Engine::new("two", &[(0, &p0), (1, &p1)]).await;
    let limited = TrustedKvAwarePolicy::with_config(TrustedKvAwareConfig {
        max_sources: 1,
        ..Default::default()
    });
    assert!(limited
        .runtime
        .discover(&reqwest::Client::new(), &two.url, None)
        .await
        .unwrap_err()
        .contains("capacity"));
}

#[tokio::test]
async fn duplicate_replay_endpoint_is_rejected_before_credit() {
    let replay_owner = Publisher::with_replay();
    let mut duplicate = Publisher::new();
    duplicate.replay_endpoint = replay_owner.replay_endpoint.clone();
    let engine = Engine::new("duplicate-replay", &[(0, &replay_owner), (1, &duplicate)]).await;
    let policy = policy();
    assert!(policy
        .runtime
        .discover(&reqwest::Client::new(), &engine.url, None)
        .await
        .unwrap_err()
        .contains("independently addressable"));
    let decision = policy
        .select_with_tokens(&[engine.worker()], Some("m"), Some(&[1, 2, 3, 4, 5]), true)
        .unwrap()
        .1;
    assert_eq!(decision.mode, KvRoutingMode::Fallback);
}

#[tokio::test]
async fn failed_discovery_releases_its_capacity_reservation() {
    let app = axum::Router::new().route(
        "/get_server_info",
        get(|| async { StatusCode::INTERNAL_SERVER_ERROR }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let failed_url = format!("http://{}", listener.local_addr().unwrap());
    let failed = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let publisher = Publisher::new();
    let healthy = Engine::new("healthy", &[(0, &publisher)]).await;
    let policy = TrustedKvAwarePolicy::with_config(TrustedKvAwareConfig {
        max_sources: 1,
        ..Default::default()
    });
    assert!(policy
        .runtime
        .discover(&reqwest::Client::new(), &failed_url, None)
        .await
        .is_err());
    policy
        .runtime
        .discover(&reqwest::Client::new(), &healthy.url, None)
        .await
        .unwrap();
    publisher.subscribed().await;
    failed.abort();
}

#[path = "common/mock_vllm_rs.rs"]
mod mock_vllm_rs;
use vllm_router_rs::backend::pb;

#[derive(Clone)]
struct ControlStub {
    instance: String,
    endpoints: Vec<String>,
}
#[tonic::async_trait]
impl pb::control_server::Control for ControlStub {
    async fn get_server_info(
        &self,
        _request: tonic::Request<pb::GetServerInfoRequest>,
    ) -> Result<tonic::Response<pb::ServerInfo>, tonic::Status> {
        Ok(tonic::Response::new(pb::ServerInfo {
            instance_id: self.instance.clone(),
            kv_block_size: 2,
            ..Default::default()
        }))
    }
    async fn get_model_info(
        &self,
        _request: tonic::Request<pb::GetModelInfoRequest>,
    ) -> Result<tonic::Response<pb::ModelInfo>, tonic::Status> {
        Ok(tonic::Response::new(pb::ModelInfo {
            model_id: "m".into(),
            served_model_name: "m".into(),
            supports_token_ids_input: true,
            ..Default::default()
        }))
    }
    async fn abort(
        &self,
        _request: tonic::Request<pb::AbortRequest>,
    ) -> Result<tonic::Response<pb::AbortResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("unused test RPC"))
    }
    async fn load_lora(
        &self,
        _request: tonic::Request<pb::LoadLoraRequest>,
    ) -> Result<tonic::Response<pb::LoadLoraResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("unused test RPC"))
    }
    async fn unload_lora(
        &self,
        _request: tonic::Request<pb::UnloadLoraRequest>,
    ) -> Result<tonic::Response<pb::UnloadLoraResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("unused test RPC"))
    }
    async fn list_loras(
        &self,
        _request: tonic::Request<pb::ListLorasRequest>,
    ) -> Result<tonic::Response<pb::ListLorasResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("unused test RPC"))
    }
    async fn get_kv_event_sources(
        &self,
        _request: tonic::Request<pb::GetKvEventSourcesRequest>,
    ) -> Result<tonic::Response<pb::GetKvEventSourcesResponse>, tonic::Status> {
        Ok(tonic::Response::new(pb::GetKvEventSourcesResponse {
            sources: self
                .endpoints
                .iter()
                .enumerate()
                .map(|(rank, endpoint)| pb::KvEventSource {
                    data_parallel_rank: Some(rank as u32),
                    endpoint: endpoint.clone(),
                    topic: "kv".into(),
                    transport: "zmq".into(),
                    ..Default::default()
                })
                .collect(),
        }))
    }
    async fn pause_generation(
        &self,
        _request: tonic::Request<pb::PauseGenerationRequest>,
    ) -> Result<tonic::Response<pb::PauseGenerationResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("unused test RPC"))
    }
    async fn resume_generation(
        &self,
        _request: tonic::Request<pb::ResumeGenerationRequest>,
    ) -> Result<tonic::Response<pb::ResumeGenerationResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("unused test RPC"))
    }
    async fn is_paused(
        &self,
        _request: tonic::Request<pb::IsPausedRequest>,
    ) -> Result<tonic::Response<pb::IsPausedResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("unused test RPC"))
    }
    async fn sleep(
        &self,
        _request: tonic::Request<pb::SleepRequest>,
    ) -> Result<tonic::Response<pb::SleepResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("unused test RPC"))
    }
    async fn wake_up(
        &self,
        _request: tonic::Request<pb::WakeUpRequest>,
    ) -> Result<tonic::Response<pb::WakeUpResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("unused test RPC"))
    }
    async fn is_sleeping(
        &self,
        _request: tonic::Request<pb::IsSleepingRequest>,
    ) -> Result<tonic::Response<pb::IsSleepingResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("unused test RPC"))
    }
    async fn init_weight_transfer_engine(
        &self,
        _request: tonic::Request<pb::InitWeightTransferEngineRequest>,
    ) -> Result<tonic::Response<pb::InitWeightTransferEngineResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("unused test RPC"))
    }
    async fn start_weight_update(
        &self,
        _request: tonic::Request<pb::StartWeightUpdateRequest>,
    ) -> Result<tonic::Response<pb::StartWeightUpdateResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("unused test RPC"))
    }
    async fn start_draft_weight_update(
        &self,
        _request: tonic::Request<pb::StartDraftWeightUpdateRequest>,
    ) -> Result<tonic::Response<pb::StartDraftWeightUpdateResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("unused test RPC"))
    }
    async fn update_weights(
        &self,
        _request: tonic::Request<pb::UpdateWeightsRequest>,
    ) -> Result<tonic::Response<pb::UpdateWeightsResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("unused test RPC"))
    }
    async fn finish_weight_update(
        &self,
        _request: tonic::Request<pb::FinishWeightUpdateRequest>,
    ) -> Result<tonic::Response<pb::FinishWeightUpdateResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("unused test RPC"))
    }
    async fn update_weight_version(
        &self,
        _request: tonic::Request<pb::UpdateWeightVersionRequest>,
    ) -> Result<tonic::Response<pb::UpdateWeightVersionResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("unused test RPC"))
    }
    async fn get_weight_version(
        &self,
        _request: tonic::Request<pb::GetWeightVersionRequest>,
    ) -> Result<tonic::Response<pb::GetWeightVersionResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("unused test RPC"))
    }
}

#[derive(Clone)]
struct InferenceStub {
    engine: mock_vllm_rs::MockVllmRs,
    ranks: Arc<Mutex<Vec<String>>>,
}
#[tonic::async_trait]
impl pb::inference_server::Inference for InferenceStub {
    type GenerateStreamStream =
        <mock_vllm_rs::MockVllmRs as pb::inference_server::Inference>::GenerateStreamStream;
    async fn generate(
        &self,
        request: tonic::Request<pb::GenerateRequest>,
    ) -> Result<tonic::Response<pb::GenerateResponse>, tonic::Status> {
        pb::inference_server::Inference::generate(&self.engine, request).await
    }
    async fn generate_stream(
        &self,
        request: tonic::Request<pb::GenerateRequest>,
    ) -> Result<tonic::Response<Self::GenerateStreamStream>, tonic::Status> {
        self.ranks.lock().unwrap().push(
            request
                .metadata()
                .get("x-data-parallel-rank")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("absent")
                .to_string(),
        );
        pb::inference_server::Inference::generate_stream(&self.engine, request).await
    }
}

#[tokio::test]
async fn grpc_control_discovery_and_prepared_tokens_select_the_holding_rank_on_the_wire() {
    let p0 = Publisher::new();
    let p1 = Publisher::new();
    let engine = mock_vllm_rs::MockVllmRs::default();
    let ranks = Arc::new(Mutex::new(Vec::new()));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("grpc://{}", listener.local_addr().unwrap());
    let (reporter, health) = tonic_health::server::health_reporter();
    reporter
        .set_service_status("", tonic_health::ServingStatus::Serving)
        .await;
    let inference = pb::inference_server::InferenceServer::new(InferenceStub {
        engine: engine.clone(),
        ranks: ranks.clone(),
    });
    let control = pb::control_server::ControlServer::new(ControlStub {
        instance: "native-dp".into(),
        endpoints: vec![p0.endpoint.clone(), p1.endpoint.clone()],
    });
    let task = tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(health)
            .add_service(inference)
            .add_service(control)
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
            .await
            .unwrap();
    });
    let config = RouterConfig {
        mode: RoutingMode::Regular {
            worker_urls: vec![url.clone()],
        },
        policy: PolicyConfig::TrustedKvAware {
            config: TrustedKvAwareConfig {
                cache_weight: 0.9,
                ..Default::default()
            },
        },
        intra_node_data_parallel_size: 2,
        disable_retries: true,
        worker_startup_timeout_secs: 3,
        worker_startup_check_interval_secs: 1,
        ..Default::default()
    };
    let ctx =
        Arc::new(AppContext::new(config, reqwest::Client::new(), 64, None, Vec::new()).unwrap());
    let router = Router::new(vec![url], &ctx).await.unwrap();
    router.pin_test_token_ids(vec![1, 2, 3, 4, 5]);
    p0.subscribed().await;
    p1.subscribed().await;
    p0.send(0, 1, vec![clear()]);
    p1.send(1, 1, vec![clear(), stored()]);
    let policy = ctx.policy_registry.get_default_policy();
    let kv = policy
        .as_any()
        .downcast_ref::<TrustedKvAwarePolicy>()
        .unwrap();
    wait_for(kv, &ctx.worker_registry.get_all(), KvRoutingMode::Exact).await;
    let request: vllm_router_rs::protocols::spec::ChatCompletionRequest = serde_json::from_value(
        json!({"model":"m","messages":[{"role":"user","content":"hello"}],"max_tokens":1}),
    )
    .unwrap();
    let response = router
        .route_typed_request(None, &request, "/v1/chat/completions", Some("m"))
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["x-router-kv-mode"], "exact");
    let _ = axum::body::to_bytes(response.into_body(), 16384)
        .await
        .unwrap();
    let captures = engine.captured.lock().unwrap();
    assert_eq!(captures.len(), 1);
    assert_eq!(captures[0].token_ids, vec![1, 2, 3, 4, 5]);
    assert!(!captures[0].had_text_prompt);
    assert_eq!(*ranks.lock().unwrap(), vec!["1"]);
    assert!(ctx.worker_registry.get_all().iter().all(|w| w.load() == 0));
    task.abort();
}

fn snapshot_now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis()
        .try_into()
        .unwrap()
}

fn snapshot_identity() -> PortableCacheIdentity {
    PortableCacheIdentity::new("m", "tokenizer-v1", PortableHashMode::Sha256, 2)
}

fn snapshot_fence(owner_id: &str, epoch: u64) -> PortableSnapshotFence {
    PortableSnapshotFence::new(owner_id, epoch)
}

fn snapshot_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn portable_snapshot_for(
    snapshot_id: &str,
    targets: &[(&str, u32)],
    incarnation: u64,
    sequence: u64,
    created_at_unix_ms: u64,
    observed_at_unix_ms: u64,
) -> PortableKvSnapshot {
    let locals = local_hashes(&[1, 2, 3, 4], 2, None);
    let sources = targets
        .iter()
        .enumerate()
        .map(
            |(source_index, (instance_id, dp_rank))| PortableSourceSnapshot {
                instance_id: (*instance_id).into(),
                dp_rank: *dp_rank,
                incarnation,
                sequence,
                observed_at_unix_ms,
                residencies: vec![PortableResidency {
                    owner: PortableResidencyOwner::Worker {
                        instance_id: (*instance_id).into(),
                        dp_rank: *dp_rank,
                        incarnation,
                    },
                    tier: PortableStorageTier::Device,
                    prefixes: vec![PortablePrefix {
                        parent_block_hash: None,
                        blocks: locals
                            .iter()
                            .enumerate()
                            .map(|(block_index, local_hash)| PortableBlock {
                                block_hash: format!(
                                    "b:{source_index}:{incarnation}:{sequence}:{block_index}"
                                ),
                                local_hash: snapshot_hex(local_hash),
                            })
                            .collect(),
                    }],
                }],
            },
        )
        .collect();
    let mut snapshot = PortableKvSnapshot {
        format_version: PORTABLE_SNAPSHOT_FORMAT_VERSION,
        snapshot_id: snapshot_id.into(),
        complete: true,
        cache_identity: snapshot_identity(),
        created_at_unix_ms,
        sources,
        fence: snapshot_fence("router-a", 1),
        checksum_sha256: String::new(),
    };
    snapshot.seal().unwrap();
    snapshot
}

#[tokio::test]
async fn portable_snapshot_applies_exactly_is_idempotent_and_rejects_id_reuse() {
    let publisher = Publisher::new();
    let engine = Engine::new("snapshot-holder", &[(0, &publisher)]).await;
    let policy = policy();
    attach(&policy, &engine, &[&publisher]).await;
    let now = snapshot_now_ms();
    let snapshot = portable_snapshot_for(
        "snapshot-a",
        &[("snapshot-holder", 0)],
        3,
        7,
        now,
        now.saturating_sub(100),
    );

    assert_eq!(
        policy
            .runtime
            .import_portable_snapshot(
                &snapshot,
                &snapshot_identity(),
                &snapshot_fence("router-a", 1),
            )
            .unwrap(),
        vllm_router_rs::kv_index::PortableSnapshotImport {
            snapshot_id: "snapshot-a".into(),
            status: PortableSnapshotImportStatus::Applied,
            sources: 1,
            blocks: 2,
        }
    );
    let decision = policy
        .select_with_tokens(&[engine.worker()], Some("m"), Some(&[1, 2, 3, 4, 5]), true)
        .unwrap()
        .1;
    assert_eq!(decision.mode, KvRoutingMode::Exact);
    assert_eq!(decision.scores[0].credited_blocks, 2.0);

    let repeated = policy
        .runtime
        .import_portable_snapshot(
            &snapshot,
            &snapshot_identity(),
            &snapshot_fence("router-a", 1),
        )
        .unwrap();
    assert_eq!(
        repeated.status,
        PortableSnapshotImportStatus::AlreadyApplied
    );

    let next = portable_snapshot_for("snapshot-b", &[("snapshot-holder", 0)], 3, 8, now, now);
    policy
        .runtime
        .import_portable_snapshot(&next, &snapshot_identity(), &snapshot_fence("router-a", 1))
        .unwrap();

    let mut same_sequence_conflict = next.clone();
    same_sequence_conflict.snapshot_id = "snapshot-conflict".into();
    same_sequence_conflict.sources[0].residencies[0].prefixes[0].blocks[0].local_hash =
        "different-local-hash".into();
    same_sequence_conflict.seal().unwrap();
    assert_eq!(
        policy
            .runtime
            .import_portable_snapshot(
                &same_sequence_conflict,
                &snapshot_identity(),
                &snapshot_fence("router-a", 1),
            )
            .unwrap_err(),
        PortableSnapshotError::SnapshotConflict
    );

    let mut reused = snapshot.clone();
    reused.sources[0].sequence = 9;
    reused.seal().unwrap();
    assert_eq!(
        policy
            .runtime
            .import_portable_snapshot(
                &reused,
                &snapshot_identity(),
                &snapshot_fence("router-a", 1),
            )
            .unwrap_err(),
        PortableSnapshotError::SnapshotIdConflict
    );
    let decision = policy
        .select_with_tokens(&[engine.worker()], Some("m"), Some(&[1, 2, 3, 4, 5]), true)
        .unwrap()
        .1;
    assert_eq!(decision.mode, KvRoutingMode::Exact);
    assert_eq!(decision.scores[0].credited_blocks, 2.0);
}

#[tokio::test]
async fn portable_snapshot_fences_advance_without_cross_epoch_ownership() {
    let publisher = Publisher::new();
    let engine = Engine::new("fenced-holder", &[(0, &publisher)]).await;
    let policy = policy();
    attach(&policy, &engine, &[&publisher]).await;
    let now = snapshot_now_ms();

    let baseline = portable_snapshot_for(
        "snapshot-fence-baseline",
        &[("fenced-holder", 0)],
        1,
        7,
        now,
        now,
    );
    policy
        .runtime
        .import_portable_snapshot(
            &baseline,
            &snapshot_identity(),
            &snapshot_fence("router-a", 1),
        )
        .unwrap();

    assert_eq!(
        policy
            .runtime
            .import_portable_snapshot(
                &baseline,
                &snapshot_identity(),
                &snapshot_fence("router-b", 1),
            )
            .unwrap_err(),
        PortableSnapshotError::FenceMismatch
    );

    let mut owner_change = portable_snapshot_for(
        "snapshot-fence-owner-change",
        &[("fenced-holder", 0)],
        1,
        8,
        now,
        now,
    );
    owner_change.fence = snapshot_fence("router-b", 1);
    owner_change.seal().unwrap();
    assert_eq!(
        policy
            .runtime
            .import_portable_snapshot(
                &owner_change,
                &snapshot_identity(),
                &snapshot_fence("router-b", 1),
            )
            .unwrap_err(),
        PortableSnapshotError::EpochOwnerConflict
    );

    let mut failed_advance = portable_snapshot_for(
        "snapshot-fence-failed-advance",
        &[("fenced-holder", 0), ("unexpected-holder", 0)],
        1,
        8,
        now,
        now,
    );
    failed_advance.fence = snapshot_fence("router-b", 2);
    failed_advance.seal().unwrap();
    assert_eq!(
        policy
            .runtime
            .import_portable_snapshot(
                &failed_advance,
                &snapshot_identity(),
                &snapshot_fence("router-b", 2),
            )
            .unwrap_err(),
        PortableSnapshotError::SourceSetMismatch
    );

    let still_epoch_one = portable_snapshot_for(
        "snapshot-fence-still-one",
        &[("fenced-holder", 0)],
        1,
        8,
        now,
        now,
    );
    policy
        .runtime
        .import_portable_snapshot(
            &still_epoch_one,
            &snapshot_identity(),
            &snapshot_fence("router-a", 1),
        )
        .unwrap();

    let mut next_epoch = portable_snapshot_for(
        "snapshot-fence-epoch-two",
        &[("fenced-holder", 0)],
        1,
        9,
        now,
        now,
    );
    next_epoch.fence = snapshot_fence("router-b", 2);
    next_epoch.seal().unwrap();
    assert_eq!(
        policy
            .runtime
            .import_portable_snapshot(
                &next_epoch,
                &snapshot_identity(),
                &snapshot_fence("router-b", 2),
            )
            .unwrap()
            .status,
        PortableSnapshotImportStatus::Applied
    );

    let stale = portable_snapshot_for(
        "snapshot-fence-stale",
        &[("fenced-holder", 0)],
        1,
        10,
        now,
        now,
    );
    assert_eq!(
        policy
            .runtime
            .import_portable_snapshot(&stale, &snapshot_identity(), &snapshot_fence("router-a", 1),)
            .unwrap_err(),
        PortableSnapshotError::StaleEpoch
    );

    assert_eq!(
        policy
            .runtime
            .import_portable_snapshot(
                &next_epoch,
                &snapshot_identity(),
                &snapshot_fence("router-b", 2),
            )
            .unwrap()
            .status,
        PortableSnapshotImportStatus::AlreadyApplied
    );
}

#[tokio::test]
async fn portable_snapshot_pins_tokenizer_identity_to_the_discovered_pool() {
    let publisher = Publisher::new();
    let engine = Engine::new("tokenizer-holder", &[(0, &publisher)]).await;
    let policy = policy();
    attach(&policy, &engine, &[&publisher]).await;
    let now = snapshot_now_ms();

    let baseline = portable_snapshot_for(
        "snapshot-tokenizer-v1",
        &[("tokenizer-holder", 0)],
        1,
        7,
        now,
        now,
    );
    policy
        .runtime
        .import_portable_snapshot(
            &baseline,
            &snapshot_identity(),
            &snapshot_fence("router-a", 1),
        )
        .unwrap();

    let other_tokenizer =
        PortableCacheIdentity::new("m", "tokenizer-v2", PortableHashMode::Sha256, 2);
    let mut changed_tokenizer = portable_snapshot_for(
        "snapshot-tokenizer-v2",
        &[("tokenizer-holder", 0)],
        1,
        8,
        now,
        now,
    );
    changed_tokenizer.cache_identity = other_tokenizer.clone();
    changed_tokenizer.seal().unwrap();

    assert_eq!(
        policy
            .runtime
            .import_portable_snapshot(
                &changed_tokenizer,
                &other_tokenizer,
                &snapshot_fence("router-a", 1),
            )
            .unwrap_err(),
        PortableSnapshotError::IdentityMismatch
    );

    let decision = policy
        .select_with_tokens(&[engine.worker()], Some("m"), Some(&[1, 2, 3, 4, 5]), true)
        .unwrap()
        .1;
    assert_eq!(decision.mode, KvRoutingMode::Exact);
    assert_eq!(decision.scores[0].credited_blocks, 2.0);
}

#[tokio::test]
async fn portable_snapshot_fence_authority_survives_runtime_restart() {
    let first_publisher = Publisher::new();
    let first_engine = Engine::new("restart-holder", &[(0, &first_publisher)]).await;
    let authority = Arc::new(SharedFenceAuthority::default());
    let first = TrustedKvAwarePolicy {
        runtime: Arc::new(
            vllm_router_rs::kv_index::runtime::KvRuntime::with_snapshot_fence_authority(
                TrustedKvAwareConfig {
                    cache_weight: 0.9,
                    ..Default::default()
                },
                Some(authority.clone()),
            ),
        ),
    };
    attach(&first, &first_engine, &[&first_publisher]).await;
    let now = snapshot_now_ms();

    let mut epoch_two = portable_snapshot_for(
        "snapshot-authority-epoch-two",
        &[("restart-holder", 0)],
        1,
        7,
        now,
        now,
    );
    epoch_two.fence = snapshot_fence("router-b", 2);
    epoch_two.seal().unwrap();
    first
        .runtime
        .import_portable_snapshot(
            &epoch_two,
            &snapshot_identity(),
            &snapshot_fence("router-b", 2),
        )
        .unwrap();

    let restarted_publisher = Publisher::new();
    let restarted_engine = Engine::new("restart-holder", &[(0, &restarted_publisher)]).await;
    let restarted = TrustedKvAwarePolicy {
        runtime: Arc::new(
            vllm_router_rs::kv_index::runtime::KvRuntime::with_snapshot_fence_authority(
                TrustedKvAwareConfig {
                    cache_weight: 0.9,
                    ..Default::default()
                },
                Some(authority),
            ),
        ),
    };
    attach(&restarted, &restarted_engine, &[&restarted_publisher]).await;

    let stale = portable_snapshot_for(
        "snapshot-authority-stale",
        &[("restart-holder", 0)],
        1,
        8,
        now,
        now,
    );
    assert_eq!(
        restarted
            .runtime
            .import_portable_snapshot(&stale, &snapshot_identity(), &snapshot_fence("router-a", 1),)
            .unwrap_err(),
        PortableSnapshotError::StaleEpoch
    );
    assert_eq!(
        restarted
            .select_with_tokens(
                &[restarted_engine.worker()],
                Some("m"),
                Some(&[1, 2, 3, 4, 5]),
                true,
            )
            .unwrap()
            .1
            .mode,
        KvRoutingMode::Fallback
    );

    let mut owner_change = portable_snapshot_for(
        "snapshot-authority-owner-change",
        &[("restart-holder", 0)],
        1,
        8,
        now,
        now,
    );
    owner_change.fence = snapshot_fence("router-a", 2);
    owner_change.seal().unwrap();
    assert_eq!(
        restarted
            .runtime
            .import_portable_snapshot(
                &owner_change,
                &snapshot_identity(),
                &snapshot_fence("router-a", 2),
            )
            .unwrap_err(),
        PortableSnapshotError::EpochOwnerConflict
    );

    let mut invalid_advance = portable_snapshot_for(
        "snapshot-authority-invalid-advance",
        &[("restart-holder", 0), ("unexpected-holder", 0)],
        1,
        8,
        now,
        now,
    );
    invalid_advance.fence = snapshot_fence("router-c", 3);
    invalid_advance.seal().unwrap();
    assert_eq!(
        restarted
            .runtime
            .import_portable_snapshot(
                &invalid_advance,
                &snapshot_identity(),
                &snapshot_fence("router-c", 3),
            )
            .unwrap_err(),
        PortableSnapshotError::SourceSetMismatch
    );

    let mut current = portable_snapshot_for(
        "snapshot-authority-current",
        &[("restart-holder", 0)],
        1,
        8,
        now,
        now,
    );
    current.fence = snapshot_fence("router-b", 2);
    current.seal().unwrap();
    assert_eq!(
        restarted
            .runtime
            .import_portable_snapshot(
                &current,
                &snapshot_identity(),
                &snapshot_fence("router-b", 2),
            )
            .unwrap()
            .status,
        PortableSnapshotImportStatus::Applied
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn snapshot_authority_commit_does_not_hold_the_live_state_lock() {
    let publisher = Publisher::new();
    let engine = Engine::new("authority-lock", &[(0, &publisher)]).await;
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let authority = Arc::new(BlockingFenceAuthority {
        current: Mutex::new(None),
        entered: entered_tx,
        release: Mutex::new(Some(release_rx)),
    });
    let policy = TrustedKvAwarePolicy {
        runtime: Arc::new(
            vllm_router_rs::kv_index::runtime::KvRuntime::with_snapshot_fence_authority(
                TrustedKvAwareConfig {
                    cache_weight: 0.9,
                    ..Default::default()
                },
                Some(authority),
            ),
        ),
    };
    attach(&policy, &engine, &[&publisher]).await;

    let now = snapshot_now_ms();
    let snapshot = portable_snapshot_for(
        "snapshot-authority-lock",
        &[("authority-lock", 0)],
        1,
        7,
        now,
        now,
    );
    let import_runtime = policy.runtime.clone();
    let import_snapshot = snapshot.clone();
    let import_task = tokio::task::spawn_blocking(move || {
        import_runtime.import_portable_snapshot(
            &import_snapshot,
            &snapshot_identity(),
            &snapshot_fence("router-a", 1),
        )
    });
    entered_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("authority should be called");

    let (mutation_done_tx, mutation_done_rx) = tokio::sync::oneshot::channel();
    let mutation_runtime = policy.runtime.clone();
    let engine_url = engine.url.clone();
    tokio::task::spawn_blocking(move || {
        mutation_runtime.remove_worker(&engine_url);
        let _ = mutation_done_tx.send(());
    });
    let mutation_completed = tokio::time::timeout(Duration::from_secs(1), mutation_done_rx)
        .await
        .is_ok();
    release_tx.send(()).unwrap();

    assert!(
        mutation_completed,
        "live state mutation blocked behind the fence authority"
    );
    assert_eq!(
        import_task.await.unwrap().unwrap_err(),
        PortableSnapshotError::ConcurrentSnapshotMutation
    );

    // The authority accepted this fence before the race was detected. Replaying
    // the same fence must therefore be idempotent and commit the retry.
    let retry_publisher = Publisher::new();
    let retry_engine = Engine::new("authority-lock", &[(0, &retry_publisher)]).await;
    attach(&policy, &retry_engine, &[&retry_publisher]).await;
    assert_eq!(
        policy
            .runtime
            .import_portable_snapshot(
                &snapshot,
                &snapshot_identity(),
                &snapshot_fence("router-a", 1),
            )
            .unwrap()
            .status,
        PortableSnapshotImportStatus::Applied
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn snapshot_authority_ignores_unrelated_pool_mutation() {
    let target_publisher = Publisher::new();
    let target_engine = Engine::new("authority-target", &[(0, &target_publisher)]).await;
    let other_publisher = Publisher::new();
    let other_engine =
        Engine::with_cache("authority-other", &[(0, &other_publisher)], "other", 2).await;
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let authority = Arc::new(BlockingFenceAuthority {
        current: Mutex::new(None),
        entered: entered_tx,
        release: Mutex::new(Some(release_rx)),
    });
    let policy = TrustedKvAwarePolicy {
        runtime: Arc::new(
            vllm_router_rs::kv_index::runtime::KvRuntime::with_snapshot_fence_authority(
                TrustedKvAwareConfig {
                    cache_weight: 0.9,
                    ..Default::default()
                },
                Some(authority),
            ),
        ),
    };
    attach(&policy, &target_engine, &[&target_publisher]).await;
    attach(&policy, &other_engine, &[&other_publisher]).await;

    let now = snapshot_now_ms();
    let snapshot = portable_snapshot_for(
        "snapshot-authority-independent",
        &[("authority-target", 0)],
        1,
        7,
        now,
        now,
    );
    let import_runtime = policy.runtime.clone();
    let import_snapshot = snapshot.clone();
    let import_task = tokio::task::spawn_blocking(move || {
        import_runtime.import_portable_snapshot(
            &import_snapshot,
            &snapshot_identity(),
            &snapshot_fence("router-a", 1),
        )
    });
    entered_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("authority should be called");

    let (mutation_done_tx, mutation_done_rx) = tokio::sync::oneshot::channel();
    let mutation_runtime = policy.runtime.clone();
    let other_engine_url = other_engine.url.clone();
    tokio::task::spawn_blocking(move || {
        mutation_runtime.remove_worker(&other_engine_url);
        let _ = mutation_done_tx.send(());
    });
    assert!(
        tokio::time::timeout(Duration::from_secs(1), mutation_done_rx)
            .await
            .is_ok(),
        "unrelated pool mutation blocked behind the fence authority"
    );
    release_tx.send(()).unwrap();

    assert_eq!(
        import_task.await.unwrap().unwrap().status,
        PortableSnapshotImportStatus::Applied,
        "an unrelated pool mutation must not invalidate the staged target"
    );
}

#[tokio::test]
async fn stale_snapshot_establishes_topology_but_never_awards_exact_credit() {
    let publisher = Publisher::new();
    let engine = Engine::new("stale-holder", &[(0, &publisher)]).await;
    let policy = TrustedKvAwarePolicy::with_config(TrustedKvAwareConfig {
        cache_weight: 0.9,
        max_event_age_secs: 1,
        ..Default::default()
    });
    attach(&policy, &engine, &[&publisher]).await;
    let now = snapshot_now_ms();
    let snapshot = portable_snapshot_for(
        "snapshot-stale",
        &[("stale-holder", 0)],
        1,
        4,
        now.saturating_sub(5_000),
        now.saturating_sub(5_000),
    );
    policy
        .runtime
        .import_portable_snapshot(
            &snapshot,
            &snapshot_identity(),
            &snapshot_fence("router-a", 1),
        )
        .unwrap();

    let decision = policy
        .select_with_tokens(&[engine.worker()], Some("m"), Some(&[1, 2, 3, 4, 5]), true)
        .unwrap()
        .1;
    assert_eq!(decision.mode, KvRoutingMode::Fallback);
    assert_eq!(decision.fallback_reason, Some("stale"));
    assert_eq!(decision.scores[0].credited_blocks, 0.0);
}

#[tokio::test]
async fn failed_snapshot_imports_preserve_the_previous_residency_and_trust() {
    let publisher = Publisher::new();
    let engine = Engine::new("atomic-holder", &[(0, &publisher)]).await;
    let policy = policy();
    attach(&policy, &engine, &[&publisher]).await;
    let now = snapshot_now_ms();
    let baseline = portable_snapshot_for(
        "snapshot-baseline",
        &[("atomic-holder", 0)],
        3,
        7,
        now.saturating_sub(1_000),
        now.saturating_sub(1_000),
    );
    policy
        .runtime
        .import_portable_snapshot(
            &baseline,
            &snapshot_identity(),
            &snapshot_fence("router-a", 1),
        )
        .unwrap();

    let mut tampered = baseline.clone();
    tampered.snapshot_id = "snapshot-tampered".into();
    tampered.sources[0].sequence += 1;
    assert_eq!(
        policy
            .runtime
            .import_portable_snapshot(
                &tampered,
                &snapshot_identity(),
                &snapshot_fence("router-a", 1),
            )
            .unwrap_err(),
        PortableSnapshotError::ChecksumMismatch
    );

    let obsolete = portable_snapshot_for(
        "snapshot-obsolete",
        &[("atomic-holder", 0)],
        2,
        20,
        now,
        now,
    );
    assert_eq!(
        policy
            .runtime
            .import_portable_snapshot(
                &obsolete,
                &snapshot_identity(),
                &snapshot_fence("router-a", 1),
            )
            .unwrap_err(),
        PortableSnapshotError::ObsoleteIncarnation
    );

    let sequence_rollback = portable_snapshot_for(
        "snapshot-sequence-rollback",
        &[("atomic-holder", 0)],
        3,
        6,
        now,
        now,
    );
    assert_eq!(
        policy
            .runtime
            .import_portable_snapshot(
                &sequence_rollback,
                &snapshot_identity(),
                &snapshot_fence("router-a", 1),
            )
            .unwrap_err(),
        PortableSnapshotError::Rollback
    );

    let created_rollback = portable_snapshot_for(
        "snapshot-created-rollback",
        &[("atomic-holder", 0)],
        3,
        8,
        now.saturating_sub(2_000),
        now.saturating_sub(2_000),
    );
    assert_eq!(
        policy
            .runtime
            .import_portable_snapshot(
                &created_rollback,
                &snapshot_identity(),
                &snapshot_fence("router-a", 1),
            )
            .unwrap_err(),
        PortableSnapshotError::Rollback
    );

    let observed_rollback = portable_snapshot_for(
        "snapshot-observed-rollback",
        &[("atomic-holder", 0)],
        3,
        8,
        now,
        now.saturating_sub(2_000),
    );
    assert_eq!(
        policy
            .runtime
            .import_portable_snapshot(
                &observed_rollback,
                &snapshot_identity(),
                &snapshot_fence("router-a", 1),
            )
            .unwrap_err(),
        PortableSnapshotError::Rollback
    );

    let decision = policy
        .select_with_tokens(&[engine.worker()], Some("m"), Some(&[1, 2, 3, 4, 5]), true)
        .unwrap()
        .1;
    assert_eq!(decision.mode, KvRoutingMode::Exact);
    assert_eq!(decision.scores[0].credited_blocks, 2.0);
}

#[tokio::test]
async fn snapshot_identity_and_source_set_must_match_the_discovered_pool() {
    let publisher = Publisher::new();
    let engine = Engine::new("expected-source", &[(0, &publisher)]).await;
    let policy = policy();
    attach(&policy, &engine, &[&publisher]).await;
    let now = snapshot_now_ms();

    let missing = portable_snapshot_for("snapshot-missing", &[], 1, 1, now, now);
    assert_eq!(
        policy
            .runtime
            .import_portable_snapshot(
                &missing,
                &snapshot_identity(),
                &snapshot_fence("router-a", 1),
            )
            .unwrap_err(),
        PortableSnapshotError::SourceSetMismatch
    );

    let extra = portable_snapshot_for(
        "snapshot-extra",
        &[("expected-source", 0), ("unexpected-source", 0)],
        1,
        1,
        now,
        now,
    );
    assert_eq!(
        policy
            .runtime
            .import_portable_snapshot(&extra, &snapshot_identity(), &snapshot_fence("router-a", 1),)
            .unwrap_err(),
        PortableSnapshotError::SourceSetMismatch
    );

    let valid = portable_snapshot_for(
        "snapshot-identity",
        &[("expected-source", 0)],
        1,
        1,
        now,
        now,
    );
    for identity in [
        PortableCacheIdentity::new("m", "other-tokenizer", PortableHashMode::Sha256, 2),
        PortableCacheIdentity::new("other-model", "tokenizer-v1", PortableHashMode::Sha256, 2),
        PortableCacheIdentity::new("m", "tokenizer-v1", PortableHashMode::XxhashCbor, 2),
        PortableCacheIdentity::new("m", "tokenizer-v1", PortableHashMode::Sha256, 4),
    ] {
        assert_eq!(
            policy
                .runtime
                .import_portable_snapshot(&valid, &identity, &snapshot_fence("router-a", 1),)
                .unwrap_err(),
            PortableSnapshotError::IdentityMismatch
        );
    }
}

#[tokio::test]
async fn shared_owner_snapshot_does_not_fabricate_device_residency() {
    let publisher = Publisher::new();
    let engine = Engine::new("shared-source", &[(0, &publisher)]).await;
    let policy = policy();
    attach(&policy, &engine, &[&publisher]).await;
    let now = snapshot_now_ms();
    let mut snapshot =
        portable_snapshot_for("snapshot-shared", &[("shared-source", 0)], 1, 1, now, now);
    snapshot.sources[0].residencies[0].owner = PortableResidencyOwner::CacheOwner {
        pool_id: "pool-a".into(),
        source_instance_id: "shared-source".into(),
        dp_rank: 0,
    };
    snapshot.sources[0].residencies[0].tier = PortableStorageTier::HostPinned;
    snapshot.seal().unwrap();
    policy
        .runtime
        .import_portable_snapshot(
            &snapshot,
            &snapshot_identity(),
            &snapshot_fence("router-a", 1),
        )
        .unwrap();

    let decision = policy
        .select_with_tokens(&[engine.worker()], Some("m"), Some(&[1, 2, 3, 4, 5]), true)
        .unwrap()
        .1;
    assert_eq!(decision.mode, KvRoutingMode::Exact);
    assert_eq!(decision.scores[0].credited_blocks, 0.0);
}
