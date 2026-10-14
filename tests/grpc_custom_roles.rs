//! Issue #318 regression: real rendering/encoding, local gRPC mock, no model weights.
mod common;
#[path = "common/grpc_frontend_fixture.rs"]
mod fixture;

use axum::body::to_bytes;
use common::mock_vllm_rs::MockVllmRsServer;
use fixture::model_fixture;
use serde_json::json;
use vllm_router_rs::backend::EngineFrontend;
use vllm_router_rs::protocols::spec::ChatCompletionRequest;

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
