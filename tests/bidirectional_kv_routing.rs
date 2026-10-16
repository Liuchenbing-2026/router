//! Real TCP protocol tests; workers are mocks, not a model or a physical KV transport.
use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use vllm_router_rs::{
    config::{PolicyConfig, RouterConfig, RoutingMode},
    routers::{RouterFactory, RouterTrait},
};

type Seen = Arc<Mutex<Vec<(String, Value)>>>;

async fn prefill(State(seen): State<Seen>, Json(body): Json<Value>) -> Json<Value> {
    seen.lock().unwrap().push(("P".into(), body));
    Json(
        json!({"choices":[],"kv_transfer_params":{"remote_engine_id":"P","do_remote_prefill":true,"do_remote_decode":false}}),
    )
}
async fn decode(State(seen): State<Seen>, Json(body): Json<Value>) -> axum::response::Response {
    use axum::response::IntoResponse;
    seen.lock().unwrap().push(("D".into(), body.clone()));
    let prompt = body.get("prompt").cloned().unwrap_or_else(|| {
        if body["messages"][0]["content"] == "first" {
            json!([1, 2])
        } else {
            json!([1, 2, 3, 4, 9])
        }
    });
    let params = json!({"do_remote_decode":true,"do_remote_prefill":false,"remote_engine_id":"D-epoch-1","remote_request_id":"turn-D","remote_host":"untrusted-advertisement","remote_port":5600,"remote_block_ids":[[11,12]],"remote_num_tokens":prompt.as_array().unwrap().len()+1,"remote_blocks_expiry_time":123456.0,"transfer_mode":"pull"});
    let mut reply = json!({"choices":[{"index":0,"text":"AB","message":{"role":"assistant","content":"AB"},"token_ids":[3,4],"finish_reason":"length"}],"kv_transfer_params":params});
    if body.get("messages").is_some() {
        reply["prompt_token_ids"] = prompt;
    } else {
        reply["choices"][0]["prompt_token_ids"] = prompt;
    }
    if body["stream"] == true {
        let text = format!("data: {reply}\n\ndata: [DONE]\n\n");
        let chunks: Vec<_> = text
            .as_bytes()
            .chunks(7)
            .map(|b| Ok::<_, std::io::Error>(bytes::Bytes::copy_from_slice(b)))
            .collect();
        axum::response::Response::builder()
            .header("content-type", "text/event-stream")
            .body(axum::body::Body::from_stream(futures_util::stream::iter(
                chunks,
            )))
            .unwrap()
    } else {
        Json(reply).into_response()
    }
}
async fn tokenize(State(seen): State<Seen>, Json(body): Json<Value>) -> Json<Value> {
    seen.lock().unwrap().push(("tokenize".into(), body.clone()));
    let ids = if body["messages"][0]["content"] == "rewritten" {
        json!([88, 2, 3, 4, 9])
    } else {
        json!([1, 2, 3, 4, 9])
    };
    Json(json!({"tokens":ids,"count":5,"max_model_len":8192}))
}
async fn fixture() -> (Box<dyn RouterTrait>, Seen, tokio::task::JoinHandle<()>) {
    let seen: Seen = Arc::new(Mutex::new(Vec::new()));
    let app = Router::new()
        .route("/p/health", get(|| async { "ok" }))
        .route("/d/health", get(|| async { "ok" }))
        .route("/p/v1/completions", post(prefill))
        .route("/d/v1/completions", post(decode))
        .route("/p/v1/chat/completions", post(prefill))
        .route("/d/v1/chat/completions", post(decode))
        .route("/p/tokenize", post(tokenize))
        .with_state(seen.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let config = RouterConfig {
        mode: RoutingMode::VllmPrefillDecode {
            prefill_urls: vec![(format!("http://{addr}/p"), None)],
            decode_urls: vec![format!("http://{addr}/d")],
            prefill_policy: None,
            decode_policy: None,
            discovery_address: None,
        },
        policy: PolicyConfig::RoundRobin,
        worker_startup_timeout_secs: 5,
        ..RouterConfig::default()
    };
    // JSON keeps this test buildable against the unmodified baseline, which ignores the new field.
    let mut config = serde_json::to_value(config).unwrap();
    config["bidirectional_kv"] = json!({});
    let config: RouterConfig = serde_json::from_value(config).unwrap();
    let context = Arc::new(
        vllm_router_rs::server::AppContext::new(config, reqwest::Client::new(), 64, None, vec![])
            .unwrap(),
    );
    let router = RouterFactory::create_router(&context).await.unwrap();
    (router, seen, task)
}

#[tokio::test]
async fn completion_turn_two_reuses_decode_metadata_and_rewrite_falls_back() {
    let (router, seen, task) = fixture().await;
    for (ids, expected) in [
        (vec![1, 2], "miss"),
        (vec![1, 2, 3, 4, 9], "candidate"),
        (vec![99, 2, 3, 4, 9, 3, 4, 8], "prefix_mismatch"),
    ] {
        let request=serde_json::from_value(json!({"model":"m","prompt":ids,"max_tokens":2,"return_token_ids":true,"conversation_id":"same","stream":true})).unwrap();
        let response = router.route_completion(None, &request, Some("m")).await;
        assert!(response.status().is_success());
        assert_eq!(
            response
                .headers()
                .get("x-router-kv-reuse")
                .and_then(|v| v.to_str().ok()),
            Some(expected)
        );
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        assert!(bytes.ends_with(b"data: [DONE]\n\n"));
    }
    let seen = seen.lock().unwrap();
    let p: Vec<_> = seen
        .iter()
        .filter(|(phase, _)| phase == "P")
        .map(|(_, body)| body)
        .collect();
    assert!(p[0]["kv_transfer_params"]["remote_engine_id"].is_null());
    assert_eq!(p[1]["kv_transfer_params"]["remote_engine_id"], "D-epoch-1");
    assert!(p[2]["kv_transfer_params"]["remote_engine_id"].is_null());
    assert!(seen.iter().all(|(_, v)| v.get("conversation_id").is_none()));
    task.abort();
}

#[tokio::test]
async fn chat_uses_native_tokenize_and_checks_rendered_prefix() {
    let (router, seen, task) = fixture().await;
    for (text, expected) in [
        ("first", "miss"),
        ("next", "candidate"),
        ("rewritten", "prefix_mismatch"),
    ] {
        let request=serde_json::from_value(json!({"model":"m","messages":[{"role":"user","content":text}],"chat_template_kwargs":{"enable_thinking":false},"max_tokens":2,"return_token_ids":true,"conversation_id":"chat"})).unwrap();
        let response = router.route_chat(None, &request, Some("m")).await;
        assert!(response.status().is_success());
        assert_eq!(
            response
                .headers()
                .get("x-router-kv-reuse")
                .and_then(|v| v.to_str().ok()),
            Some(expected)
        );
        response.into_body().collect().await.unwrap();
    }
    let seen = seen.lock().unwrap();
    let t: Vec<_> = seen
        .iter()
        .filter(|(phase, _)| phase == "tokenize")
        .map(|(_, body)| body)
        .collect();
    assert_eq!(t.len(), 2);
    assert_eq!(t[0]["chat_template_kwargs"]["enable_thinking"], false);
    let p: Vec<_> = seen
        .iter()
        .filter(|(phase, _)| phase == "P")
        .map(|(_, body)| body)
        .collect();
    assert_eq!(p[1]["kv_transfer_params"]["remote_engine_id"], "D-epoch-1");
    assert!(p[2]["kv_transfer_params"]["remote_engine_id"].is_null());
    task.abort();
}
