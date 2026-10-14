//! gRPC chat rendering and tokenization via `vllm-chat` and `vllm-tokenizer`.
//! Loaded models share an optional L0 encoding cache.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Result};
use dashmap::DashMap;
use parking_lot::RwLock;

use super::encode_cache::FrontendCache;
use super::tokenize_cache::{cache_key, CachedTokenize, TokenizeCache, TokenizeCacheConfig};
use super::vllm_frontend::{
    chat_request_from_openai, system_text, tool_text, user_text, OpenAiChatRequest, VllmFrontend,
};
use crate::protocols::spec::{
    ChatCompletionRequest, ChatMessage as SpecChatMessage, ToolChoice as SpecToolChoice,
    ToolChoiceValue, UserMessageContent,
};
use crate::tokenizer::{TokenizerCacheConfig, TokenizerCacheStats};
use vllm_chat::{
    AssistantContentBlock, AssistantToolCall, ChatTool, ChatToolChoice,
    ReasoningEffort as VllmReasoningEffort,
};

#[derive(Clone)]
enum Frontend {
    Vllm(Arc<VllmFrontend>),
    /// Integration-test pin: fake ids, still not this repo’s tokenizer.
    TestIds(Vec<u32>),
}

/// Loads each model once and optionally shares an L0 budget across models.
#[derive(Clone)]
pub struct TokenizerCache {
    pinned: Arc<RwLock<Option<Frontend>>>,
    by_model: Arc<DashMap<String, Arc<VllmFrontend>>>,
    load_lock: Arc<tokio::sync::Mutex<()>>,
    encoding_cache: Option<Arc<FrontendCache>>,
    /// Exact render+encode results for requests that already went through this
    /// path. Shared by every model, because the key carries the request's own
    /// model field.
    tokenize: Arc<TokenizeCache>,
}

impl std::fmt::Debug for TokenizerCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenizerCache").finish()
    }
}

impl Default for TokenizerCache {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenizerCache {
    pub fn new() -> Self {
        Self {
            pinned: Arc::new(RwLock::new(None)),
            by_model: Arc::new(DashMap::new()),
            load_lock: Arc::new(tokio::sync::Mutex::new(())),
            encoding_cache: None,
            tokenize: Arc::new(
                TokenizeCache::new(TokenizeCacheConfig::default())
                    .expect("the built-in tokenize cache budgets are valid"),
            ),
        }
    }

    /// Enable L0 with a shared budget across models.
    pub fn with_encoding_cache(config: TokenizerCacheConfig) -> Result<Self> {
        Ok(Self {
            encoding_cache: Some(Arc::new(FrontendCache::new(config)?)),
            ..Self::new()
        })
    }

    /// L0 stats across models, or None when disabled.
    pub fn encoding_cache_stats(&self) -> Option<TokenizerCacheStats> {
        self.encoding_cache.as_ref().map(|cache| cache.stats())
    }

    /// Tests only: bypass model loading by returning these fake prompt ids.
    pub fn pin_test_token_ids(&self, token_ids: Vec<u32>) {
        *self.pinned.write() = Some(Frontend::TestIds(token_ids));
    }

    pub async fn resolve(&self, model: Option<&str>) -> Result<FrontendHandle> {
        if let Some(pinned) = self.pinned.read().clone() {
            return Ok(self.handle(pinned));
        }
        let (key, source) = model_resolution(model)?;
        if let Some(hit) = self.by_model.get(&key) {
            return Ok(self.handle(Frontend::Vllm(hit.clone())));
        }
        // Loading a tokenizer is expensive. Re-check under a single-flight
        // lock so a cold burst does not load the same model N times.
        let _load_guard = self.load_lock.lock().await;
        if let Some(hit) = self.by_model.get(&key) {
            return Ok(self.handle(Frontend::Vllm(hit.clone())));
        }
        let frontend = Arc::new(
            VllmFrontend::load(&source, Default::default(), self.encoding_cache.clone())
                .await
                .map_err(|e| anyhow!("vllm-chat load {source} for key {key}: {e}"))?,
        );
        self.by_model.insert(key, frontend.clone());
        Ok(self.handle(Frontend::Vllm(frontend)))
    }

    /// Wraps a resolved frontend with the shared tokenize cache.
    fn handle(&self, frontend: Frontend) -> FrontendHandle {
        FrontendHandle {
            frontend,
            tokenize: Arc::clone(&self.tokenize),
        }
    }
}

fn model_resolution(request_model: Option<&str>) -> Result<(String, String)> {
    let request_model = request_model
        .map(str::trim)
        .filter(|model| !model.is_empty());
    let tokenizer_override = std::env::var("VLLM_ROUTER_TOKENIZER")
        .ok()
        .filter(|value| !value.is_empty())
        .map(|tok| {
            let path = Path::new(&tok);
            if path.is_dir() {
                return Ok(tok);
            }
            if path.is_file() {
                return path
                    .parent()
                    .filter(|parent| parent.is_dir())
                    .map(|parent| parent.to_string_lossy().into_owned())
                    .ok_or_else(|| anyhow!("VLLM_ROUTER_TOKENIZER has no model directory: {tok}"));
            }
            Err(anyhow!(
                "VLLM_ROUTER_TOKENIZER is not a local model directory or tokenizer file: {tok}"
            ))
        })
        .transpose()?;
    let model_override = std::env::var("VLLM_ROUTER_MODEL")
        .ok()
        .filter(|value| !value.is_empty());
    let source = tokenizer_override
        .or(model_override)
        .or_else(|| request_model.map(str::to_string))
        .ok_or_else(|| anyhow!("grpc worker requires request.model or VLLM_ROUTER_MODEL"))?;
    // Environment variables select where to load frontend assets from; the
    // request model remains the cache identity so aliases do not collapse into
    // one global key in multi-model routing.
    let key = request_model
        .map(str::to_string)
        .unwrap_or_else(|| source.clone());
    Ok((key, source))
}

#[derive(Clone)]
pub struct FrontendHandle {
    frontend: Frontend,
    /// Exact render+encode results, shared with every other handle of this
    /// `TokenizerCache` so a warm session stays warm across model aliases.
    tokenize: Arc<TokenizeCache>,
}

impl FrontendHandle {
    pub fn tokenizer(&self) -> Option<super::vllm_frontend::DynTokenizer> {
        match &self.frontend {
            Frontend::Vllm(frontend) => Some(frontend.tokenizer()),
            Frontend::TestIds(_) => None,
        }
    }

    pub fn default_temperature(&self) -> Result<Option<f32>> {
        match &self.frontend {
            Frontend::Vllm(frontend) => frontend.default_temperature(),
            Frontend::TestIds(_) => Ok(None),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TokenizeOut {
    pub token_ids: Arc<[u32]>,
    pub adapt_ms: f64,
    pub template_ms: f64,
    pub encode_ms: f64,
    pub encode_backend: &'static str,
}

impl TokenizeOut {
    pub fn frontend_ms(&self) -> f64 {
        self.adapt_ms + self.template_ms + self.encode_ms
    }
}

pub fn tokenize_chat_request_timed(
    request: &ChatCompletionRequest,
    frontend: &FrontendHandle,
) -> Result<TokenizeOut> {
    match &frontend.frontend {
        Frontend::TestIds(ids) => {
            if ids.is_empty() {
                return Err(anyhow!("pinned test token_ids are empty"));
            }
            Ok(TokenizeOut {
                token_ids: Arc::from(ids.clone()),
                adapt_ms: 0.0,
                template_ms: 0.0,
                encode_ms: 0.0,
                encode_backend: "test-pin",
            })
        }
        Frontend::Vllm(vllm) => tokenize_vllm(request, vllm, &frontend.tokenize),
    }
}

fn tokenize_vllm(
    request: &ChatCompletionRequest,
    frontend: &VllmFrontend,
    cache: &TokenizeCache,
) -> Result<TokenizeOut> {
    if request
        .tools
        .as_ref()
        .is_some_and(|tools| !tools.is_empty())
        && !matches!(
            request.tool_choice,
            Some(SpecToolChoice::Value(ToolChoiceValue::None))
        )
    {
        return Err(anyhow!(
            "active tool calling requires vllm-chat output parsing, which the gRPC response adapter does not yet expose; tool history and tool_choice=none are supported"
        ));
    }
    if request.reasoning_effort.is_some() {
        return Err(anyhow!(
            "reasoning_effort requires vllm-chat output parsing, which the gRPC response adapter does not yet expose"
        ));
    }
    // The cache key covers the whole wire request, so a hit is valid for every
    // field the lowering below reads. Looking it up before the lowering also
    // skips the OpenAI -> ChatRequest conversion, which is part of the cost of
    // a repeated agent turn.
    // The encode step needs these two fields; both live in `other`, which the
    // cache key covers, so a hit is guaranteed to match the render it returns.
    let add_special_tokens = request
        .other
        .get("add_special_tokens")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let cache_salt = request
        .other
        .get("cache_salt")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);

    let cache_key = cache_key(request)?;
    if let Some(hit) = cache.get(&cache_key) {
        // Render is skipped, but the encode still goes through the shared L0
        // encoding cache so its hit/miss accounting stays faithful.
        let (token_ids, encode_ms) = frontend
            .encode_prompt_timed(hit.prompt, add_special_tokens, cache_salt.as_deref())
            .map_err(|e| anyhow!("vllm-chat tokenize: {e}"))?;
        if token_ids.is_empty() {
            return Err(anyhow!("vllm-chat produced empty token_ids"));
        }
        return Ok(TokenizeOut {
            token_ids: Arc::from(token_ids),
            adapt_ms: 0.0,
            template_ms: hit.template_ms,
            encode_ms,
            encode_backend: "vllm-tokenizer",
        });
    }
    let t0 = Instant::now();
    let kwargs = request.chat_template_kwargs.clone().unwrap_or_default();
    if request.other.contains_key("chat_template") {
        return Err(anyhow!(
            "per-request chat_template is not supported by the router gRPC frontend"
        ));
    }
    let documents = request
        .other
        .get("documents")
        .cloned()
        .map(serde_json::from_value)
        .transpose()
        .map_err(|error| anyhow!("invalid documents: {error}"))?;
    let priority = request
        .other
        .get("priority")
        .and_then(serde_json::Value::as_i64)
        .map(i32::try_from)
        .transpose()
        .map_err(|_| anyhow!("priority is outside the i32 range"))?
        .unwrap_or(0);
    let data_parallel_rank = request
        .other
        .get("data_parallel_rank")
        .and_then(serde_json::Value::as_u64)
        .map(u32::try_from)
        .transpose()
        .map_err(|_| anyhow!("data_parallel_rank is outside the u32 range"))?;
    let session_id = request
        .session_params
        .as_ref()
        .and_then(|params| params.get("session_id"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    let upstream = chat_request_from_openai(OpenAiChatRequest {
        request_id: format!("router-{}", request.model.as_deref().unwrap_or("model")),
        messages: spec_messages_to_upstream(&request.messages)?,
        add_generation_prompt: request.add_generation_prompt,
        continue_final_message: request.continue_final_message,
        template_kwargs: kwargs,
        tools: spec_tools_to_upstream(request)?,
        tool_choice: spec_tool_choice_to_upstream(request.tool_choice.as_ref())?,
        parallel_tool_calls: request.parallel_tool_calls.unwrap_or(true),
        reasoning_effort: spec_reasoning_effort(request)?,
        response_format: request
            .response_format
            .as_ref()
            .map(serde_json::to_value)
            .transpose()?,
        add_special_tokens,
        documents,
        priority,
        cache_salt: cache_salt.clone(),
        data_parallel_rank,
        session_id,
    })?;
    let adapt_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let (prompt, template_ms) = frontend
        .chat_render_timed(&upstream)
        .map_err(|e| anyhow!("vllm-chat tokenize: {e}"))?;
    // Keep a copy for the render cache only once the whole request succeeds.
    let cached_prompt = prompt.clone();
    let (token_ids, encode_ms) = frontend
        .encode_prompt_timed(
            prompt,
            upstream.add_special_tokens,
            upstream.cache_salt.as_deref(),
        )
        .map_err(|e| anyhow!("vllm-chat tokenize: {e}"))?;
    if token_ids.is_empty() {
        return Err(anyhow!("vllm-chat produced empty token_ids"));
    }
    let token_ids: Arc<[u32]> = Arc::from(token_ids);
    cache.put(
        cache_key,
        CachedTokenize {
            prompt: cached_prompt,
            template_ms,
        },
    );
    Ok(TokenizeOut {
        token_ids,
        adapt_ms,
        template_ms,
        encode_ms,
        encode_backend: "vllm-tokenizer",
    })
}

fn spec_messages_to_upstream(
    messages: &[SpecChatMessage],
) -> Result<Vec<super::vllm_frontend::UpstreamChatMessage>> {
    messages
        .iter()
        .map(|message| {
            Ok(match message {
                SpecChatMessage::System { content, .. } => system_text(user_content_text(content)?),
                SpecChatMessage::User { content, .. } => user_text(user_content_text(content)?),
                SpecChatMessage::Assistant {
                    content,
                    tool_calls,
                    reasoning,
                    ..
                } => {
                    let mut blocks = Vec::new();
                    if let Some(reasoning) = reasoning.as_ref().filter(|text| !text.is_empty()) {
                        blocks.push(AssistantContentBlock::Reasoning {
                            text: reasoning.clone(),
                        });
                    }
                    if let Some(content) = content {
                        blocks.push(AssistantContentBlock::Text {
                            text: user_content_text(content)?,
                        });
                    }
                    if let Some(tool_calls) = tool_calls {
                        for call in tool_calls {
                            if call.tool_type != "function" {
                                return Err(anyhow!(
                                    "only function assistant tool calls are supported"
                                ));
                            }
                            blocks.push(AssistantContentBlock::ToolCall(AssistantToolCall {
                                id: call.id.clone(),
                                name: call.function.name.clone(),
                                arguments: call
                                    .function
                                    .arguments
                                    .clone()
                                    .unwrap_or_else(|| "{}".to_string()),
                            }));
                        }
                    }
                    if blocks.is_empty() {
                        return Err(anyhow!(
                            "assistant message must contain content, reasoning, or tool_calls"
                        ));
                    }
                    super::vllm_frontend::UpstreamChatMessage::assistant_blocks(blocks)
                }
                SpecChatMessage::Tool {
                    content,
                    tool_call_id,
                    ..
                } => {
                    let text = content.as_str().ok_or_else(|| {
                        anyhow!("tool message content must be a string on the gRPC path")
                    })?;
                    tool_text(text, tool_call_id)
                }
                SpecChatMessage::Function { .. } => {
                    return Err(anyhow!(
                        "legacy function messages are not supported by vllm-chat"
                    ));
                }
                SpecChatMessage::Other(value) => {
                    // HTTP backends can forward unknown roles verbatim, but vllm-chat
                    // cannot represent them, so the gRPC frontend rejects them.
                    let role = value
                        .get("role")
                        .and_then(serde_json::Value::as_str)
                        .expect("ChatMessage::Other must contain a string `role`");
                    return Err(anyhow!(
                        "chat message role `{role}` is not supported by the router gRPC frontend"
                    ));
                }
            })
        })
        .collect()
}

fn spec_tools_to_upstream(request: &ChatCompletionRequest) -> Result<Vec<ChatTool>> {
    request
        .tools
        .as_deref()
        .unwrap_or_default()
        .iter()
        .map(|tool| {
            if tool.tool_type != "function" {
                return Err(anyhow!("only function tools are supported"));
            }
            Ok(ChatTool {
                name: tool.function.name.clone(),
                description: tool.function.description.clone(),
                parameters: tool.function.parameters.clone(),
                strict: None,
            })
        })
        .collect()
}

fn spec_tool_choice_to_upstream(choice: Option<&SpecToolChoice>) -> Result<Option<ChatToolChoice>> {
    choice
        .map(|choice| {
            Ok(match choice {
                SpecToolChoice::Value(ToolChoiceValue::Auto) => ChatToolChoice::Auto,
                SpecToolChoice::Value(ToolChoiceValue::Required) => ChatToolChoice::Required,
                SpecToolChoice::Value(ToolChoiceValue::None) => ChatToolChoice::None,
                SpecToolChoice::Function {
                    tool_type,
                    function,
                } => {
                    if tool_type != "function" {
                        return Err(anyhow!("only function tool_choice is supported"));
                    }
                    ChatToolChoice::Function {
                        name: function.name.clone(),
                    }
                }
            })
        })
        .transpose()
}

fn spec_reasoning_effort(request: &ChatCompletionRequest) -> Result<Option<VllmReasoningEffort>> {
    request
        .reasoning_effort
        .as_ref()
        .map(|effort| {
            let value = serde_json::to_value(effort)?;
            serde_json::from_value(value).map_err(Into::into)
        })
        .transpose()
}

fn user_content_text(content: &UserMessageContent) -> Result<String> {
    match content {
        UserMessageContent::Text(text) => Ok(text.clone()),
        UserMessageContent::Parts(parts) => {
            let mut text = String::new();
            for part in parts {
                match part {
                    crate::protocols::spec::ContentPart::Text { text: part } => {
                        text.push_str(part);
                    }
                    crate::protocols::spec::ContentPart::ImageUrl { .. } => {
                        return Err(anyhow!(
                            "multimodal chat is not supported by the token_ids-only gRPC path"
                        ));
                    }
                }
            }
            Ok(text)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    fn model_dir() -> Option<String> {
        let dir = std::env::var("VLLM_ROUTER_MODEL")
            .ok()
            .filter(|s| !s.is_empty())?;
        Path::new(&dir)
            .join("tokenizer.json")
            .is_file()
            .then_some(dir)
    }

    fn python_bin() -> PathBuf {
        if let Ok(py) = std::env::var("PYTHON") {
            if !py.is_empty() {
                return PathBuf::from(py);
            }
        }
        if let Ok(venv) = std::env::var("VIRTUAL_ENV") {
            let cand = PathBuf::from(venv).join("bin/python");
            if cand.is_file() {
                return cand;
            }
        }
        PathBuf::from("python3")
    }

    fn python_vllm_chat_ids(
        model: &str,
        messages: &serde_json::Value,
    ) -> Result<Option<Vec<u32>>, String> {
        let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/python_vllm_chat_ids.py");
        let out = Command::new(python_bin())
            .arg(&script)
            .arg(model)
            .arg(messages.to_string())
            .output()
            .map_err(|e| format!("spawn python: {e}"))?;
        if !out.status.success() {
            let err = String::from_utf8_lossy(&out.stderr);
            if err.contains("ModuleNotFoundError") || err.contains("No module named") {
                return Ok(None);
            }
            return Err(format!(
                "python vllm tokenize failed ({:?}): {err}",
                out.status
            ));
        }
        let stdout = String::from_utf8_lossy(&out.stdout);
        let ids = serde_json::from_slice(&out.stdout).or_else(|first_error| {
            stdout
                .lines()
                .rev()
                .map(str::trim)
                .find(|line| line.starts_with('['))
                .ok_or(first_error)
                .and_then(serde_json::from_str)
        });
        ids.map(Some)
            .map_err(|e| format!("python stdout not a json id list: {e}; stdout={}", stdout))
    }

    async fn rust_vllm_chat_ids(model: &str, messages: serde_json::Value) -> Vec<u32> {
        let frontend = VllmFrontend::load(model, Default::default(), None)
            .await
            .expect("vllm-chat load_model_backends");
        let req: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": model,
            "messages": messages,
            "max_tokens": 8,
            "add_generation_prompt": true
        }))
        .unwrap();
        // A throwaway cache: this probe renders the request once and has no
        // interest in the router's shared entries.
        let cache = TokenizeCache::new(TokenizeCacheConfig::default())
            .expect("the built-in tokenize cache budgets are valid");
        tokenize_vllm(&req, &frontend, &cache)
            .expect("vllm-chat tokenize")
            .token_ids
            .to_vec()
    }

    #[test]
    fn preserves_reasoning_and_assistant_tool_history() {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "messages": [{
                "role": "assistant",
                "reasoning": "check the weather",
                "content": "calling tool",
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "arguments": "{\"city\":\"Paris\"}"
                    }
                }]
            }]
        }))
        .unwrap();

        let messages = spec_messages_to_upstream(&request.messages).unwrap();
        let vllm_chat::ChatMessage::Assistant { content } = &messages[0] else {
            panic!("expected assistant message");
        };
        assert!(matches!(
            &content[0],
            AssistantContentBlock::Reasoning { text } if text == "check the weather"
        ));
        assert!(matches!(
            &content[1],
            AssistantContentBlock::Text { text } if text == "calling tool"
        ));
        assert!(matches!(
            &content[2],
            AssistantContentBlock::ToolCall(call)
                if call.id == "call-1"
                    && call.name == "weather"
                    && call.arguments == "{\"city\":\"Paris\"}"
        ));
    }

    #[test]
    fn rejects_unknown_roles_on_grpc_path() {
        for role in ["latest_reminder", "developer"] {
            let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
                "messages": [{
                    "role": role,
                    "content": "Model-specific instructions."
                }]
            }))
            .unwrap();

            let error = spec_messages_to_upstream(&request.messages).unwrap_err();
            assert_eq!(
                error.to_string(),
                format!("chat message role `{role}` is not supported by the router gRPC frontend")
            );
        }
    }

    #[test]
    fn preserves_request_tools_and_named_choice() {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "lookup weather",
                    "parameters": {"type": "object"}
                }
            }],
            "tool_choice": {
                "type": "function",
                "function": {"name": "weather"}
            }
        }))
        .unwrap();

        let messages = spec_messages_to_upstream(&request.messages).unwrap();
        let tools = spec_tools_to_upstream(&request).unwrap();
        let choice = spec_tool_choice_to_upstream(request.tool_choice.as_ref()).unwrap();
        let chat = chat_request_from_openai(OpenAiChatRequest {
            request_id: "request-1".into(),
            messages,
            add_generation_prompt: request.add_generation_prompt,
            continue_final_message: request.continue_final_message,
            template_kwargs: Default::default(),
            tools,
            tool_choice: choice,
            parallel_tool_calls: true,
            reasoning_effort: None,
            response_format: None,
            add_special_tokens: false,
            documents: None,
            priority: 0,
            cache_salt: None,
            data_parallel_rank: None,
            session_id: None,
        })
        .unwrap();
        assert_eq!(chat.tools().len(), 1);
        assert_eq!(chat.tools()[0].name, "weather");
        assert!(matches!(
            chat.tool_choice(),
            ChatToolChoice::Function { name } if name == "weather"
        ));
    }

    #[tokio::test]
    async fn vllm_chat_tokenize_matches_python_vllm_user() {
        let Some(model) = model_dir() else {
            eprintln!("skip: set VLLM_ROUTER_MODEL to a model dir with tokenizer.json");
            return;
        };
        let messages = serde_json::json!([{"role": "user", "content": "hello"}]);
        let Some(py_ids) = python_vllm_chat_ids(&model, &messages).expect("python vllm") else {
            eprintln!("skip: python cannot import vllm (set PYTHON or VIRTUAL_ENV)");
            return;
        };
        let rust_ids = rust_vllm_chat_ids(&model, messages).await;
        assert_eq!(
            rust_ids, py_ids,
            "vllm-chat+vllm-tokenizer must match Python vllm.tokenizers.get_tokenizer"
        );
    }

    #[tokio::test]
    async fn vllm_chat_tokenize_matches_python_vllm_system_user() {
        let Some(model) = model_dir() else {
            eprintln!("skip: set VLLM_ROUTER_MODEL to a model dir with tokenizer.json");
            return;
        };
        let messages = serde_json::json!([
            {"role": "system", "content": "You are a test."},
            {"role": "user", "content": "ping"}
        ]);
        let Some(py_ids) = python_vllm_chat_ids(&model, &messages).expect("python vllm") else {
            eprintln!("skip: python cannot import vllm (set PYTHON or VIRTUAL_ENV)");
            return;
        };
        let rust_ids = rust_vllm_chat_ids(&model, messages).await;
        assert_eq!(
            rust_ids, py_ids,
            "vllm-chat+vllm-tokenizer must match Python vllm on system+user"
        );
    }
}
