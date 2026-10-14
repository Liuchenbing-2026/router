//! Issue #318 regression: real rendering/encoding, local gRPC mock, no model weights.
mod common;
#[path = "common/grpc_frontend_fixture.rs"]
mod fixture;

use axum::body::to_bytes;
use common::mock_vllm_rs::MockVllmRsServer;
use fixture::model_fixture;
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;
use vllm_router_rs::backend::EngineFrontend;
use vllm_router_rs::config::{PolicyConfig, RouterConfig, RoutingMode};
use vllm_router_rs::protocols::spec::ChatCompletionRequest;
use vllm_router_rs::routers::http::router::Router;

async fn grpc_app(worker: &MockVllmRsServer) -> axum::Router {
    let config = RouterConfig {
        mode: RoutingMode::Regular {
            worker_urls: vec![worker.grpc_url.clone()],
        },
        policy: PolicyConfig::RoundRobin,
        disable_circuit_breaker: true,
        ..Default::default()
    };
    let router = Router::new(
        vec![worker.grpc_url.clone()],
        &common::create_test_context(config.clone()),
    )
    .await
    .unwrap();
    common::test_app::create_test_app(Arc::new(router), reqwest::Client::new(), &config)
}

fn post_chat(body: &serde_json::Value) -> axum::http::Request<axum::body::Body> {
    axum::http::Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(serde_json::to_vec(body).unwrap()))
        .unwrap()
}

#[tokio::test]
async fn custom_role_http_entry_reaches_grpc_in_streaming_and_nonstreaming_modes() {
    let model = model_fixture();
    let worker = MockVllmRsServer::spawn().await;
    let app = grpc_app(&worker).await;
    // Compute the oracle independently of EngineFrontend::prepare.
    let tokenizer = tokenizers::Tokenizer::from_file(model.path().join("tokenizer.json")).unwrap();
    let expected = tokenizer
        .encode(
            "user: Hello\nlatest_reminder: Be brief 世界\nassistant: ",
            false,
        )
        .unwrap()
        .get_ids()
        .to_vec();
    for stream in [false, true] {
        let response = app
            .clone()
            .oneshot(post_chat(&json!({
                "model": model.path().to_str().unwrap(),
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "latest_reminder", "content": "Be brief 世界"}
                ],
                "max_tokens": 8, "stream": stream
            })))
            .await
            .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        if stream {
            let text = std::str::from_utf8(&body).unwrap();
            assert!(text.contains("chat.completion.chunk"));
            assert!(text.contains("[DONE]"));
        } else {
            let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(body["usage"]["prompt_tokens"], expected.len());
        }
    }
    let captured = worker.captured();
    assert_eq!(captured.len(), 2);
    for request in captured {
        assert!(!request.had_text_prompt);
        assert_eq!(request.token_ids, expected);
    }
}

#[tokio::test]
async fn invalid_custom_messages_return_http_400_without_worker_dispatch() {
    let model = model_fixture();
    let worker = MockVllmRsServer::spawn().await;
    let app = grpc_app(&worker).await;
    for message in [
        json!({"role": "checkpoint", "content": null}),
        json!({"role": "checkpoint"}),
        json!({"role": "checkpoint", "content": "hello", "metadata": {"step": 2}}),
        json!({"role": "checkpoint", "content": [{"type": "text", "text": "hello", "extra": true}]}),
        json!({"role": "checkpoint", "content": [{"type": "image_url", "image_url": {"url": "https://example.invalid/image.png"}}]}),
    ] {
        let response = app
            .clone()
            .oneshot(post_chat(&json!({
                "model": model.path().to_str().unwrap(),
                "messages": [message], "max_tokens": 8
            })))
            .await
            .unwrap();
        let status = response.status();
        let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        assert_eq!(
            status,
            axum::http::StatusCode::BAD_REQUEST,
            "{message}: {}",
            String::from_utf8_lossy(&body)
        );
        assert!(
            worker.captured().is_empty(),
            "invalid input reached worker: {message}"
        );
    }
    // A valid request after the errors proves the worker/router remained usable.
    let response = app
        .oneshot(post_chat(&json!({
            "model": model.path().to_str().unwrap(),
            "messages": [{"role": "checkpoint", "content": "hello"}], "max_tokens": 8
        })))
        .await
        .unwrap();
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    assert_eq!(worker.captured().len(), 1);
}

#[tokio::test]
async fn custom_role_cache_separates_role_and_content_and_does_not_skip_validation() {
    let model = model_fixture();
    let frontend =
        EngineFrontend::with_tokenizer_cache(Duration::from_secs(30), Default::default()).unwrap();
    let tokenizer = tokenizers::Tokenizer::from_file(model.path().join("tokenizer.json")).unwrap();
    let mut outputs = Vec::new();
    for (role, content) in [
        ("checkpoint", "blue"),
        ("checkpoint", "red"),
        ("latest_reminder", "blue"),
    ] {
        let value = json!({
            "model": model.path().to_str().unwrap(),
            "messages": [{"role": role, "content": content}], "max_tokens": 8
        });
        let expected = tokenizer
            .encode(format!("{role}: {content}\nassistant: "), false)
            .unwrap()
            .get_ids()
            .to_vec();
        for _ in 0..2 {
            let prepared = frontend
                .prepare(serde_json::from_value(value.clone()).unwrap())
                .await
                .unwrap();
            assert_eq!(prepared.tokenized.token_ids.as_ref(), expected.as_slice());
        }
        outputs.push(expected);
        let before = frontend.tokenizer_cache_stats().unwrap();
        let mut invalid = value;
        invalid["messages"][0]["metadata"] = json!({"step": 1});
        let error = frontend
            .prepare(serde_json::from_value(invalid).unwrap())
            .await
            .err()
            .expect("cache must not bypass extension validation");
        assert!(
            error.contains("metadata") && error.contains("cannot be represented"),
            "{error}"
        );
        assert_eq!(frontend.tokenizer_cache_stats().unwrap(), before);
    }
    assert_ne!(outputs[0], outputs[1]);
    assert_ne!(outputs[0], outputs[2]);
    let stats = frontend.tokenizer_cache_stats().unwrap();
    assert_eq!((stats.misses, stats.hits, stats.entries), (3, 3, 3));
}

#[tokio::test]
async fn custom_role_reaches_grpc_worker_without_role_rewriting() {
    let model = model_fixture();
    let worker = MockVllmRsServer::spawn().await;
    let frontend = EngineFrontend::new();
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "messages": [
            {"role": "user", "content": "Summarize the context."},
            {"role": "latest_reminder", "content": "Keep the answer concise."}
        ],
        "max_tokens": 8
    }))
    .unwrap();
    let prepared = frontend.prepare(request.clone()).await.unwrap();
    let expected_ids = prepared.tokenized.token_ids.clone();
    assert!(!expected_ids.is_empty());
    let tokenizer = prepared.handle.tokenizer().unwrap();
    assert_eq!(expected_ids.as_ref(), tokenizer.encode(
        "user: Summarize the context.\nlatest_reminder: Keep the answer concise.\nassistant: ", false
    ).unwrap().as_slice());

    let mut rewritten = serde_json::to_value(&request).unwrap();
    rewritten["messages"][1]["role"] = json!("user");
    let rewritten = frontend
        .prepare(serde_json::from_value(rewritten).unwrap())
        .await
        .unwrap();
    assert_ne!(expected_ids, rewritten.tokenized.token_ids);

    let response = frontend.dispatch(&worker.grpc_url, prepared).await;
    assert_eq!(response.status(), 200);
    to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let captured = worker.captured();
    assert_eq!(captured.len(), 1);
    assert!(!captured[0].had_text_prompt);
    assert_eq!(captured[0].token_ids.as_slice(), expected_ids.as_ref());
}

#[tokio::test]
async fn deepseek_v4_and_v41_reminder_tokens_match_reference_prompt_and_reach_worker() {
    let worker = MockVllmRsServer::spawn().await;
    for model_type in ["deepseek_v4", "deepseek_v41"] {
        let model = model_fixture();
        std::fs::write(
            model.path().join("config.json"),
            json!({
                "model_type": model_type, "vocab_size": 293
            })
            .to_string(),
        )
        .unwrap();
        let frontend = EngineFrontend::with_tokenizer_cache(
            std::time::Duration::from_secs(30),
            Default::default(),
        )
        .unwrap();
        for assistant_history in [false, true] {
            let mut messages = vec![
                json!({"role": "user", "content": "Hello"}),
                json!({"role": "latest_reminder", "content": "Be brief 世界"}),
            ];
            if assistant_history {
                messages.push(json!({"role": "assistant", "content": "Hi there"}));
            }
            let request: ChatCompletionRequest = serde_json::from_value(json!({
                "model": model.path().to_str().unwrap(),
                "messages": messages,
                "chat_template_kwargs": {"thinking": false},
                "max_tokens": 8
            }))
            .unwrap();
            let prepared = frontend.prepare(request.clone()).await.unwrap();
            // Python DeepSeek V4/V4.1 encoders, thinking_mode="chat":
            // the assistant opener belongs BEFORE latest_reminder, not after it.
            let mut expected = String::from("<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜></think><｜latest_reminder｜>Be brief 世界");
            if assistant_history {
                expected.push_str("Hi there<｜end▁of▁sentence｜>");
            }
            let tokenizer = prepared.handle.tokenizer().unwrap();
            let expected_ids = tokenizer.encode(&expected, false).unwrap();
            assert_eq!(
                prepared.tokenized.token_ids.as_ref(),
                expected_ids.as_slice(),
                "{model_type}, assistant_history={assistant_history}"
            );
            let cached = frontend.prepare(request).await.unwrap();
            assert_eq!(cached.tokenized.token_ids.as_ref(), expected_ids.as_slice());
            let response = frontend.dispatch(&worker.grpc_url, cached).await;
            assert_eq!(response.status(), 200);
            to_bytes(response.into_body(), usize::MAX).await.unwrap();
            assert_eq!(worker.captured().last().unwrap().token_ids, expected_ids);
        }
        assert_eq!(frontend.tokenizer_cache_stats().unwrap().hits, 2);
    }
    assert_eq!(worker.captured().len(), 4);
}

#[tokio::test]
async fn native_renderer_rejects_unsupported_role_instead_of_rewriting_it() {
    let model = model_fixture();
    std::fs::write(
        model.path().join("config.json"),
        json!({
            "model_type": "deepseek_v4", "vocab_size": 293
        })
        .to_string(),
    )
    .unwrap();
    let frontend = EngineFrontend::new();
    let request = serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "messages": [{"role": "not_a_deepseek_role", "content": "hello"}],
        "max_tokens": 8
    }))
    .unwrap();
    let error = frontend
        .prepare(request)
        .await
        .err()
        .expect("must reject role");
    assert!(
        error.contains("not_a_deepseek_role") && error.contains("not supported"),
        "{error}"
    );
}

#[tokio::test]
async fn custom_role_does_not_bypass_continue_final_assistant_validation() {
    let model = model_fixture();
    let frontend = EngineFrontend::new();
    let request = serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "messages": [{"role": "latest_reminder", "content": "hello"}],
        "continue_final_message": true,
        "max_tokens": 8
    }))
    .unwrap();
    let error = frontend
        .prepare(request)
        .await
        .err()
        .expect("must reject continuation");
    assert!(
        error.contains("requires a final assistant message"),
        "{error}"
    );
}
