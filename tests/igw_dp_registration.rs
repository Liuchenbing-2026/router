//! IGW registration must preserve a worker's explicit DP identity.

use serde_json::json;
use std::sync::Arc;
use vllm_router_rs::{
    config::{PolicyConfig, RouterConfig},
    core::WorkerRegistry,
    policies::PolicyRegistry,
    protocols::worker_spec::WorkerConfigRequest,
    routers::router_manager::RouterManager,
};

fn manager() -> (RouterManager, Arc<WorkerRegistry>, Arc<PolicyRegistry>) {
    let workers = Arc::new(WorkerRegistry::new());
    let policies = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
    let manager = RouterManager::new(
        RouterConfig {
            enable_igw: true,
            ..Default::default()
        },
        reqwest::Client::new(),
        workers.clone(),
        policies.clone(),
    );
    (manager, workers, policies)
}

fn request(value: serde_json::Value) -> WorkerConfigRequest {
    serde_json::from_value(value).unwrap()
}

#[tokio::test]
async fn explicit_rank_registration_creates_a_real_dp_worker() {
    let (manager, workers, _) = manager();
    manager
        .add_worker(request(json!({
            "url": "http://127.0.0.1:8123", "model_id": "model",
            "worker_type": "prefill", "dp_rank": 1, "dp_size": 2
        })))
        .await
        .unwrap();
    let registered = workers.get_by_model("model");
    assert_eq!(registered.len(), 1);
    assert!(
        registered[0].is_dp_aware(),
        "IGW silently discarded explicit DP metadata"
    );
    assert_eq!(registered[0].dp_rank(), Some(1));
    assert_eq!(registered[0].dp_size(), Some(2));
    assert_eq!(registered[0].url(), "http://127.0.0.1:8123@1");
    assert_eq!(
        registered[0].endpoint_url("/health"),
        "http://127.0.0.1:8123/health"
    );
}

#[tokio::test]
async fn dp_readback_preserves_labels_priority_and_cost() {
    let (manager, _, _) = manager();
    let result = manager
        .add_worker(request(json!({
            "url":"http://localhost:8123/", "model_id":"model", "worker_type":"decode",
            "dp_rank":1, "dp_size":4, "priority":73, "cost":0.5, "labels":{"zone":"a"}
        })))
        .await
        .unwrap();
    let info = serde_json::to_value(result.worker.unwrap()).unwrap();
    assert_eq!(info["url"], "http://localhost:8123@1");
    assert_eq!(info["base_url"], "http://localhost:8123");
    assert_eq!(info["dp_rank"], 1);
    assert_eq!(info["dp_size"], 4);
    assert_eq!(info["priority"], 73);
    assert_eq!(info["cost"], 0.5);
    assert_eq!(info["metadata"]["zone"], "a");
}

#[tokio::test]
async fn invalid_dp_requests_do_not_register_any_worker() {
    let (manager, workers, _) = manager();
    let cases = [
        json!({"dp_rank":0}),
        json!({"dp_size":2}),
        json!({"dp_rank":0,"dp_size":0}),
        json!({"dp_rank":2,"dp_size":2}),
        json!({"dp_rank":1,"dp_size":2,"url":"grpc://host:1234"}),
        json!({"dp_rank":1,"dp_size":2,"url":"http://host:1234@1"}),
        json!({"dp_rank":1,"dp_size":2,"url":"http://host:1234?x=1"}),
        json!({"dp_rank":1,"dp_size":2,"url":"http://host:1234/path@1"}),
        json!({"dp_rank":1,"dp_size":2,"worker_type":"encode"}),
    ];
    for extra in cases {
        let mut body = json!({"url":"http://host:1234", "model_id":"model"});
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        assert!(
            manager.add_worker(request(body.clone())).await.is_err(),
            "accepted {body}"
        );
        assert_eq!(workers.stats().total_workers, 0);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn repeated_and_concurrent_rank_registration_is_idempotent() {
    let (manager, workers, policies) = manager();
    let manager = Arc::new(manager);
    let body = json!({"url":"http://localhost:8123", "model_id":"model", "dp_rank":1,"dp_size":2});
    let barrier = Arc::new(tokio::sync::Barrier::new(16));
    let mut handles = Vec::new();
    for _ in 0..16 {
        let manager = manager.clone();
        let body = body.clone();
        let barrier = barrier.clone();
        handles.push(tokio::spawn(async move {
            barrier.wait().await;
            manager.add_worker(request(body)).await.unwrap()
        }));
    }
    let first = handles.remove(0).await.unwrap().worker.unwrap();
    for h in handles {
        assert_eq!(h.await.unwrap().worker.unwrap().id, first.id);
    }
    let original = workers.get_by_url(&first.url).unwrap();
    original.increment_load();
    let repost = manager
        .add_worker(request(body))
        .await
        .unwrap()
        .worker
        .unwrap();
    assert_eq!(repost.id, first.id);
    assert!(Arc::ptr_eq(
        &original,
        &workers.get_by_url(&first.url).unwrap()
    ));
    assert_eq!(original.load(), 1);
    assert_eq!(workers.get_by_model_fast("model").len(), 1);
    manager.remove_worker_from_registry(&first.url).unwrap();
    assert!(policies.get_policy("model").is_none());
}

#[tokio::test]
async fn conflicting_rank_or_endpoint_topology_does_not_replace_live_workers() {
    let (manager, workers, _) = manager();
    let first = json!({"url":"http://localhost:8123","model_id":"model","dp_rank":0,"dp_size":2});
    manager.add_worker(request(first.clone())).await.unwrap();
    for extra in [
        json!({"dp_size":4}),
        json!({"worker_type":"decode"}),
        json!({"model_id":"other"}),
        json!({"priority":99}),
    ] {
        let mut body = first.clone();
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        assert!(manager.add_worker(request(body)).await.is_err());
    }
    let mut sibling = first;
    sibling["dp_rank"] = json!(1);
    sibling["dp_size"] = json!(4);
    assert!(manager.add_worker(request(sibling)).await.is_err());
    assert_eq!(workers.get_by_model("model").len(), 1);
    assert!(workers.get_by_model("other").is_empty());
}

#[tokio::test]
async fn rank_removal_and_endpoint_removal_keep_policy_counts_consistent() {
    let (manager, workers, policies) = manager();
    for rank in 0..3 {
        manager.add_worker(request(json!({"url":"http://localhost:8123","model_id":"model","dp_rank":rank,"dp_size":3}))).await.unwrap();
    }
    manager
        .remove_worker_from_registry("http://localhost:8123@1")
        .unwrap();
    assert!(manager.get_worker("http://localhost:8123@1").is_none());
    assert_eq!(workers.get_by_model_fast("model").len(), 2);
    assert!(policies.get_policy("model").is_some());
    manager
        .remove_worker_from_registry("http://localhost:8123")
        .unwrap();
    assert!(workers.get_by_model_fast("model").is_empty());
    assert!(policies.get_policy("model").is_none());
    assert!(manager
        .remove_worker_from_registry("http://localhost:8123")
        .is_err());
}

#[tokio::test]
async fn prefill_and_decode_register_independent_rank_sizes() {
    let (manager, workers, _) = manager();
    for (port, role, size) in [(8123, "prefill", 4), (8124, "decode", 2)] {
        for rank in 0..size {
            manager.add_worker(request(json!({"url":format!("http://localhost:{port}"),"model_id":"model","worker_type":role,"dp_rank":rank,"dp_size":size}))).await.unwrap();
        }
    }
    assert_eq!(workers.stats().prefill_workers, 4);
    assert_eq!(workers.stats().decode_workers, 2);
    assert_eq!(manager.list_workers().total, 6);
}

#[tokio::test]
async fn basic_registration_keeps_legacy_response_shape() {
    let (manager, workers, _) = manager();
    let result = manager
        .add_worker(request(
            json!({"url":"http://localhost:8123","model_id":"model"}),
        ))
        .await
        .unwrap();
    let body = serde_json::to_value(result.worker.unwrap()).unwrap();
    assert_eq!(body["url"], "http://localhost:8123");
    for key in ["dp_rank", "dp_size", "base_url"] {
        assert!(body.get(key).is_none());
    }
    assert!(!workers
        .get_by_url("http://localhost:8123")
        .unwrap()
        .is_dp_aware());
}

#[tokio::test]
async fn explicit_rank_zero_and_ipv6_are_not_treated_as_basic_workers() {
    let (manager, workers, _) = manager();
    manager
        .add_worker(request(
            json!({"url":"http://[::1]:8123","model_id":"model","dp_rank":0,"dp_size":1}),
        ))
        .await
        .unwrap();
    let w = workers.get_by_url("http://[::1]:8123@0").unwrap();
    assert_eq!(
        w.endpoint_url("/v1/completions"),
        "http://[::1]:8123/v1/completions"
    );
    assert!(w.is_dp_aware());
}

struct Server {
    url: String,
    task: tokio::task::JoinHandle<()>,
}

impl Drop for Server {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn serve(app: axum::Router) -> Server {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    Server { url, task }
}

async fn igw_server(enabled: bool) -> Server {
    use vllm_router_rs::{
        routers::{router_manager::RouterId, RouterFactory},
        server::{build_app, AppContext, AppState},
    };
    let config = RouterConfig {
        enable_igw: enabled,
        policy: PolicyConfig::ConsistentHash { virtual_nodes: 100 },
        ..Default::default()
    };
    let context = Arc::new(
        AppContext::new(config.clone(), reqwest::Client::new(), 64, None, vec![]).unwrap(),
    );
    let regular = Arc::from(
        RouterFactory::create_regular_router(&[], &context)
            .await
            .unwrap(),
    );
    let manager = Arc::new(RouterManager::new(
        config,
        context.client.clone(),
        context.worker_registry.clone(),
        context.policy_registry.clone(),
    ));
    manager.register_router(RouterId::new("http-regular".into()), regular);
    manager.register_router(
        RouterId::new("http-pd".into()),
        Arc::from(
            RouterFactory::create_vllm_pd_router(
                &[],
                &[],
                None,
                None,
                None,
                &context.router_config.policy,
                &context,
            )
            .await
            .unwrap(),
        ),
    );
    let state = Arc::new(AppState {
        router: manager.clone(),
        context,
        concurrency_queue_tx: None,
        router_manager: enabled.then_some(manager),
    });
    serve(build_app(state, 1024 * 1024, vec![], vec![], true)).await
}

#[tokio::test]
async fn rest_registration_routes_to_the_selected_rank_over_real_tcp() {
    use axum::{
        http::HeaderMap,
        routing::{get, post},
        Json, Router,
    };
    async fn completion(headers: HeaderMap) -> Json<serde_json::Value> {
        Json(
            json!({"choices":[{"message":{"role":"assistant","content":headers.get("x-data-parallel-rank").unwrap().to_str().unwrap()}}]}),
        )
    }
    let worker = serve(
        Router::new()
            .route("/v1/chat/completions", post(completion))
            .route("/health", get(|| async { "ok" })),
    )
    .await;
    let router = igw_server(true).await;
    let client = reqwest::Client::new();
    for rank in 0..2 {
        let response = client
            .post(format!("{}/workers", router.url))
            .json(&json!({"url":worker.url,"model_id":"model","dp_rank":rank,"dp_size":2}))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let info: serde_json::Value = response.json().await.unwrap();
        assert_eq!(info["worker"]["dp_rank"], rank);
    }
    let mut selected = None;
    for _ in 0..8 {
        let response=client.post(format!("{}/v1/chat/completions",router.url)).header("x-session-id","stable-session").header("x-data-parallel-rank","99").json(&json!({"model":"model","messages":[{"role":"user","content":"hello"}],"max_tokens":1})).send().await.unwrap();
        let status = response.status();
        let raw = response.text().await.unwrap();
        assert_eq!(status, reqwest::StatusCode::OK, "{raw}");
        let body: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let rank = body["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(
            rank == "0" || rank == "1",
            "client rank was not replaced: {rank}"
        );
        assert_eq!(selected.get_or_insert(rank.clone()), &rank);
    }
    let mut encoded = reqwest::Url::parse(&format!("{}/workers/", router.url)).unwrap();
    encoded
        .path_segments_mut()
        .unwrap()
        .pop_if_empty()
        .push(&format!("{}@{}", worker.url, selected.unwrap()));
    assert_eq!(
        client
            .delete(encoded.clone())
            .send()
            .await
            .unwrap()
            .status(),
        reqwest::StatusCode::OK
    );
    let body: serde_json::Value = client
        .get(format!("{}/workers", router.url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(body["total"], 1);
    encoded.path_segments_mut().unwrap().pop().push(&worker.url);
    assert_eq!(
        client.delete(encoded).send().await.unwrap().status(),
        reqwest::StatusCode::OK
    );
    let body: serde_json::Value = client
        .get(format!("{}/workers", router.url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(body["total"], 0);
}

#[tokio::test]
async fn non_igw_rest_registration_cannot_silently_ignore_dp_fields() {
    let router = igw_server(false).await;
    let response = reqwest::Client::new()
        .post(format!("{}/workers", router.url))
        .json(&json!({"url":"http://127.0.0.1:8123","model_id":"model","dp_rank":0,"dp_size":2}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["code"], "DP_REGISTRATION_REQUIRES_IGW");
}

#[tokio::test]
async fn server_info_and_health_use_the_network_endpoint_without_rank_suffix() {
    use axum::{routing::get, Json, Router};
    let backend = serve(
        Router::new()
            .route(
                "/get_server_info",
                get(|| async { Json(json!({"model_id":"discovered"})) }),
            )
            .route("/health", get(|| async { "ok" })),
    )
    .await;
    let (manager, workers, _) = manager();
    manager
        .add_worker(request(json!({"url":backend.url,"dp_rank":1,"dp_size":2})))
        .await
        .unwrap();
    let rank = workers.get_by_model("discovered").pop().unwrap();
    assert_eq!(rank.dp_rank(), Some(1));
    assert!(rank.check_health_async().await.is_ok());
}

#[tokio::test]
async fn rest_pd_registration_forwards_independent_prefill_and_decode_ranks() {
    use axum::{
        http::HeaderMap,
        routing::{get, post},
        Json, Router,
    };
    let seen = Arc::new(std::sync::Mutex::new(Vec::<(String, String)>::new()));
    let prefill_seen = seen.clone();
    let prefill=serve(Router::new().route("/health",get(||async{"ok"})).route("/v1/chat/completions",post(move |headers:HeaderMap| {
        let seen=prefill_seen.clone();async move {
            seen.lock().unwrap().push(("P".into(),headers.get("x-data-parallel-rank").unwrap().to_str().unwrap().into()));
            Json(json!({"kv_transfer_params":{"remote_engine_id":"P","do_remote_prefill":true,"remote_block_ids":[1]}}))
        }
    }))).await;
    let decode_seen = seen.clone();
    let decode=serve(Router::new().route("/health",get(||async{"ok"})).route("/v1/chat/completions",post(move |headers:HeaderMap,Json(body):Json<serde_json::Value>| {
        let seen=decode_seen.clone();async move {
            seen.lock().unwrap().push(("D".into(),headers.get("x-data-parallel-rank").unwrap().to_str().unwrap().into()));
            assert!(body["kv_transfer_params"].is_object());
            Json(json!({"choices":[{"message":{"role":"assistant","content":"ok"}}]}))
        }
    }))).await;
    let router = igw_server(true).await;
    let client = reqwest::Client::new();
    for (url, role, rank, size) in [
        (&prefill.url, "prefill", 3, 4),
        (&decode.url, "decode", 1, 2),
    ] {
        let response=client.post(format!("{}/workers",router.url)).json(&json!({"url":url,"worker_type":role,"model_id":"model","dp_rank":rank,"dp_size":size})).send().await.unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
    }
    let response = client
        .post(format!("{}/v1/chat/completions", router.url))
        .header("x-data-parallel-rank", "99")
        .json(
            &json!({"model":"model","messages":[{"role":"user","content":"hello"}],"max_tokens":1}),
        )
        .send()
        .await
        .unwrap();
    let status = response.status();
    let body = response.text().await.unwrap();
    assert_eq!(status, reqwest::StatusCode::OK, "{body}");
    assert_eq!(
        *seen.lock().unwrap(),
        vec![("P".into(), "3".into()), ("D".into(), "1".into())]
    );
}
