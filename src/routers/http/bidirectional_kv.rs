//! Bounded, single-use D→P metadata. Residency is owned by the engine, not this cache.
use crate::{config::BidirectionalKvConfig, core::Worker};
use axum::{body::Body, http::HeaderMap, response::Response};
use futures_util::StreamExt;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, Weak},
    time::{Duration, Instant},
};
use uuid::Uuid;

type Key = [u8; 32];

#[derive(Debug)]
struct Entry {
    params: Value,
    prefix_hash: Key,
    prefix_len: usize,
    source: Weak<dyn Worker>,
    expires: Instant,
    bytes: usize,
}

#[derive(Debug)]
struct Slot {
    generation: Uuid,
    in_flight: bool,
    touched: Instant,
    entry: Option<Entry>,
}

#[derive(Debug, Default)]
struct State {
    slots: HashMap<Key, Slot>,
}

#[derive(Debug)]
pub(super) struct ConversationCache {
    config: BidirectionalKvConfig,
    state: Mutex<State>,
}

fn hash_tokens(tokens: &[u32]) -> Key {
    let mut hash = Sha256::new();
    for token in tokens {
        hash.update(token.to_le_bytes());
    }
    hash.finalize().into()
}

fn tokens(value: &Value, limit: usize) -> Option<Vec<u32>> {
    let array = value.as_array()?;
    if array.len() > limit {
        return None;
    }
    array
        .iter()
        .map(|v| u32::try_from(v.as_u64()?).ok())
        .collect()
}

fn request_key(
    request: &Value,
    headers: Option<&HeaderMap>,
    conversation: &str,
    path: &str,
) -> Option<Key> {
    if conversation.is_empty() || conversation.len() > 256 {
        return None;
    }
    let model = request.get("model")?.as_str()?;
    let mut hash = Sha256::new();
    // Length delimiters prevent ambiguous concatenation. Credentials never enter logs or labels.
    for part in [
        headers
            .and_then(|h| h.get("authorization"))
            .map(|v| v.as_bytes())
            .unwrap_or_default(),
        headers
            .and_then(|h| h.get("x-api-key"))
            .map(|v| v.as_bytes())
            .unwrap_or_default(),
        headers
            .and_then(|h| h.get("x-tenant-id"))
            .map(|v| v.as_bytes())
            .unwrap_or_default(),
        model.as_bytes(),
        path.as_bytes(),
        conversation.as_bytes(),
    ] {
        hash.update(part.len().to_le_bytes());
        hash.update(part);
    }
    Some(hash.finalize().into())
}

impl ConversationCache {
    pub(super) fn new(config: BidirectionalKvConfig) -> Arc<Self> {
        Arc::new(Self {
            config,
            state: Mutex::new(State::default()),
        })
    }

    fn has_candidate(&self, key: &Key) -> bool {
        self.state
            .lock()
            .unwrap()
            .slots
            .get(key)
            .is_some_and(|s| s.entry.is_some())
    }

    fn prune(&self, state: &mut State, now: Instant) {
        let ttl = Duration::from_secs(self.config.ttl_secs);
        state
            .slots
            .retain(|_, s| s.in_flight || now.duration_since(s.touched) < ttl);
    }

    fn evict_oldest(state: &mut State, except: Option<Key>) -> bool {
        let key = state
            .slots
            .iter()
            .filter(|(k, s)| Some(**k) != except && !s.in_flight)
            .min_by_key(|(_, s)| s.touched)
            .map(|(k, _)| *k);
        if let Some(key) = key {
            state.slots.remove(&key);
            true
        } else {
            false
        }
    }

    fn claim(
        self: &Arc<Self>,
        key: Key,
        current: Option<&[u32]>,
        live: &[Arc<dyn Worker>],
    ) -> (Option<Turn>, Option<Value>, &'static str) {
        let now = Instant::now();
        let mut state = self.state.lock().unwrap();
        self.prune(&mut state, now);
        if state.slots.get(&key).is_some_and(|s| s.in_flight) {
            // Fence the old attempt and decline both competing turns; neither can publish.
            state.slots.remove(&key);
            return (None, None, "concurrent_turn");
        }
        while !state.slots.contains_key(&key) && state.slots.len() >= self.config.max_entries {
            if !Self::evict_oldest(&mut state, None) {
                return (None, None, "capacity");
            }
        }
        let old = state.slots.remove(&key).and_then(|s| s.entry);
        let mut reason = "miss";
        let candidate = old.and_then(|entry| {
            if now >= entry.expires {
                reason = "expired";
                return None;
            }
            let Some(source) = entry.source.upgrade() else {
                reason = "source_removed";
                return None;
            };
            if !source.is_healthy() || !live.iter().any(|w| Arc::ptr_eq(w, &source)) {
                reason = "source_removed";
                return None;
            }
            let Some(current) = current else {
                reason = "unverified_prefix";
                return None;
            };
            if current.len() < entry.prefix_len
                || hash_tokens(&current[..entry.prefix_len]) != entry.prefix_hash
            {
                reason = "prefix_mismatch";
                return None;
            }
            reason = "candidate";
            Some(entry.params)
        });
        let generation = Uuid::new_v4();
        state.slots.insert(
            key,
            Slot {
                generation,
                in_flight: true,
                touched: now,
                entry: None,
            },
        );
        (
            Some(Turn {
                cache: self.clone(),
                key,
                generation,
                started: now,
            }),
            candidate,
            reason,
        )
    }

    /// Token IDs are the authority. Textual history equality does not prove BPE/template equality.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn begin(
        self: &Arc<Self>,
        request: &mut Value,
        path: &str,
        headers: Option<&HeaderMap>,
        prefill_url: &str,
        client: &reqwest::Client,
        api_key: Option<&str>,
        live: &[Arc<dyn Worker>],
    ) -> (Option<Turn>, Option<Value>, &'static str) {
        let conversation = request
            .as_object_mut()
            .and_then(|o| o.remove("conversation_id"));
        let Some(conversation) = conversation.as_ref().and_then(Value::as_str) else {
            return (None, None, "no_conversation");
        };
        if !matches!(path, "/v1/completions" | "/v1/chat/completions")
            || request.get("return_token_ids") != Some(&Value::Bool(true))
            || request.get("n").and_then(Value::as_u64).unwrap_or(1) != 1
            || request.get("best_of").and_then(Value::as_u64).unwrap_or(1) != 1
            || request
                .get("echo")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            || request
                .get("truncate_prompt_tokens")
                .is_some_and(|v| !v.is_null())
            || request.get("lora_path").is_some_and(|v| !v.is_null())
            || request.get("cache_salt").is_some_and(|v| !v.is_null())
            || [
                "prompt_embeds",
                "inputs_embeds",
                "multi_modal_data",
                "mm_processor_kwargs",
                "media_io_kwargs",
                "documents",
                "functions",
                "tools",
            ]
            .iter()
            .any(|field| {
                request
                    .get(*field)
                    .is_some_and(|v| !v.is_null() && v.as_array().is_none_or(|a| !a.is_empty()))
            })
            || request
                .get("messages")
                .and_then(Value::as_array)
                .is_some_and(|messages| {
                    messages.iter().any(|m| {
                        m.get("content")
                            .is_some_and(|c| !c.is_null() && !c.is_string())
                    })
                })
            || request
                .get("prompt")
                .is_some_and(|v| !v.is_string() && tokens(v, self.config.max_tokens).is_none())
        {
            return (None, None, "unsupported_request");
        }
        let Some(key) = request_key(request, headers, conversation, path) else {
            return (None, None, "invalid_identity");
        };
        let mut current = request
            .get("prompt")
            .and_then(|v| tokens(v, self.config.max_tokens));
        if current.is_none() && self.has_candidate(&key) {
            let mut payload = serde_json::Map::new();
            // The native /tokenize schema shares these render options with generation.
            for field in [
                "model",
                "prompt",
                "messages",
                "add_generation_prompt",
                "continue_final_message",
                "add_special_tokens",
                "chat_template",
                "chat_template_kwargs",
                "tools",
            ] {
                if let Some(value) = request.get(field).filter(|v| !v.is_null()) {
                    payload.insert(field.into(), value.clone());
                }
            }
            // Restrict to textual inputs; multimodal processor identity is outside this path.
            let textual = request
                .get("messages")
                .and_then(Value::as_array)
                .is_none_or(|messages| {
                    messages.iter().all(|m| {
                        m.get("content")
                            .is_none_or(|c| c.is_null() || c.is_string())
                    })
                });
            if textual {
                let mut builder = client
                    .post(format!("{}/tokenize", prefill_url.trim_end_matches('/')))
                    .json(&payload);
                if let Some(api_key) = api_key {
                    builder = builder.bearer_auth(api_key);
                }
                if let Ok(response) = builder.send().await {
                    if response.status().is_success() {
                        // Tokenization is advisory; any failure preserves ordinary prefill.
                        let mut body = Vec::new();
                        let mut stream = response.bytes_stream();
                        let mut valid = true;
                        while let Some(chunk) = stream.next().await {
                            match chunk {
                                Ok(chunk)
                                    if body.len().saturating_add(chunk.len())
                                        <= self.config.max_entry_bytes =>
                                {
                                    body.extend_from_slice(&chunk)
                                }
                                _ => {
                                    valid = false;
                                    break;
                                }
                            }
                        }
                        if valid {
                            if let Ok(value) = serde_json::from_slice::<Value>(&body) {
                                current = tokens(&value["tokens"], self.config.max_tokens);
                            }
                        }
                    }
                }
            }
        }
        self.claim(key, current.as_deref(), live)
    }
}

#[derive(Debug)]
pub(super) struct Turn {
    cache: Arc<ConversationCache>,
    key: Key,
    generation: Uuid,
    started: Instant,
}

impl Turn {
    fn publish(&self, mut params: Value, prompt: &[u32], output: &[u32], source: &Arc<dyn Worker>) {
        if prompt.is_empty()
            || output.is_empty()
            || prompt.len() + output.len() > self.cache.config.max_tokens
        {
            return;
        }
        let Some(obj) = params.as_object_mut() else {
            return;
        };
        let Some(remote_tokens) = obj
            .get("remote_num_tokens")
            .and_then(Value::as_u64)
            .and_then(|n| usize::try_from(n).ok())
        else {
            return;
        };
        if remote_tokens == 0 || remote_tokens > prompt.len() + output.len() - 1 {
            return;
        }
        if obj.get("do_remote_decode") != Some(&Value::Bool(true))
            || obj.get("transfer_mode").is_some_and(|v| v != "pull")
            || obj
                .get("remote_engine_id")
                .is_none_or(|v| v.as_str().is_none_or(|s| s.is_empty()))
            || obj
                .get("remote_request_id")
                .is_none_or(|v| v.as_str().is_none_or(|s| s.is_empty()))
            || !obj
                .get("remote_port")
                .is_some_and(|v| v.as_u64().is_some_and(|n| n > 0 && n <= u16::MAX as u64))
            || !obj.get("remote_block_ids").is_some_and(|v| {
                v.as_array().is_some_and(|groups| {
                    !groups.is_empty()
                        && groups.iter().all(|g| {
                            g.as_array().is_some_and(|blocks| {
                                !blocks.is_empty() && blocks.iter().all(|b| b.as_u64().is_some())
                            })
                        })
                })
            })
        {
            return;
        }
        let Ok(url) = reqwest::Url::parse(source.base_url()) else {
            return;
        };
        let Some(http_host) = url.host_str() else {
            return;
        };
        // A private NIXL side channel may differ from the public HTTP/API hostname.
        // Metadata comes from the selected trusted decoder, never from the client.
        let host = obj
            .get("remote_host")
            .and_then(Value::as_str)
            .filter(|host| !host.is_empty())
            .unwrap_or(http_host)
            .to_owned();
        obj.insert("remote_host".into(), json!(host));
        obj.insert("do_remote_decode".into(), json!(true));
        obj.insert("do_remote_prefill".into(), json!(false));
        let now = Instant::now();
        // Start before dispatch, not on metadata arrival: slow decoding/transport never renews
        // an engine lease. Native remote_blocks_expiry_time is an opaque remote monotonic clock.
        let Some(expires) = self
            .started
            .checked_add(Duration::from_secs(self.cache.config.ttl_secs))
        else {
            return;
        };
        if now >= expires {
            return;
        }
        let bytes = serde_json::to_vec(&params).map_or(usize::MAX, |b| b.len());
        if bytes > self.cache.config.max_entry_bytes || bytes > self.cache.config.max_bytes {
            return;
        }
        let mut prefix = Vec::with_capacity(prompt.len() + output.len() - 1);
        prefix.extend_from_slice(prompt);
        prefix.extend_from_slice(&output[..output.len() - 1]);
        let mut state = self.cache.state.lock().unwrap();
        if !state
            .slots
            .get(&self.key)
            .is_some_and(|s| s.generation == self.generation && s.in_flight)
        {
            return;
        }
        while state
            .slots
            .values()
            .filter_map(|s| s.entry.as_ref())
            .map(|e| e.bytes)
            .sum::<usize>()
            + bytes
            > self.cache.config.max_bytes
        {
            if !ConversationCache::evict_oldest(&mut state, Some(self.key)) {
                return;
            }
        }
        if let Some(slot) = state.slots.get_mut(&self.key) {
            slot.entry = Some(Entry {
                params,
                prefix_hash: hash_tokens(&prefix[..remote_tokens]),
                prefix_len: remote_tokens,
                source: Arc::downgrade(source),
                expires,
                bytes,
            });
            slot.in_flight = false;
            slot.touched = now;
        }
    }

    pub(super) fn observe(self, response: Response, source: Arc<dyn Worker>) -> Response {
        if !response.status().is_success() {
            return response;
        }
        let streaming = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|s| s.starts_with("text/event-stream"));
        let limit = self.cache.config.max_entry_bytes;
        let max_tokens = self.cache.config.max_tokens;
        let observer = Observer {
            turn: self,
            source,
            streaming,
            limit,
            max_tokens,
            line: Vec::new(),
            event: Vec::new(),
            prompt: Vec::new(),
            output: Vec::new(),
            params: None,
            finished: false,
            done: false,
            invalid: false,
        };
        let (parts, body) = response.into_parts();
        let stream = futures_util::stream::unfold(
            (body.into_data_stream(), observer),
            |(mut stream, mut observer)| async move {
                match stream.next().await {
                    Some(Ok(bytes)) => {
                        observer.feed(&bytes);
                        Some((Ok::<_, axum::Error>(bytes), (stream, observer)))
                    }
                    Some(Err(error)) => {
                        observer.invalid = true;
                        Some((Err(error), (stream, observer)))
                    }
                    None => {
                        observer.finish();
                        None
                    }
                }
            },
        );
        Response::from_parts(parts, Body::from_stream(stream))
    }
}

impl Drop for Turn {
    fn drop(&mut self) {
        let mut state = self.cache.state.lock().unwrap();
        if state
            .slots
            .get(&self.key)
            .is_some_and(|s| s.generation == self.generation && s.in_flight)
        {
            state.slots.remove(&self.key);
        }
    }
}

struct Observer {
    turn: Turn,
    source: Arc<dyn Worker>,
    streaming: bool,
    limit: usize,
    max_tokens: usize,
    line: Vec<u8>,
    event: Vec<u8>,
    prompt: Vec<u32>,
    output: Vec<u32>,
    params: Option<Value>,
    finished: bool,
    done: bool,
    invalid: bool,
}

impl Observer {
    fn record(&mut self, value: &Value) {
        if value.get("error").is_some() {
            self.invalid = true;
            return;
        }
        if let Some(ids) = value.get("prompt_token_ids").filter(|v| !v.is_null()) {
            match tokens(ids, self.max_tokens) {
                Some(ids) if self.prompt.is_empty() || self.prompt == ids => self.prompt = ids,
                _ => self.invalid = true,
            }
        }
        if let Some(params) = value.get("kv_transfer_params").filter(|v| !v.is_null()) {
            self.params = Some(params.clone());
        }
        if let Some(choices) = value.get("choices").and_then(Value::as_array) {
            if choices.len() > 1 {
                self.invalid = true;
                return;
            }
            for choice in choices {
                if choice.get("index").and_then(Value::as_u64).unwrap_or(0) != 0 {
                    self.invalid = true;
                    return;
                }
                if let Some(ids) = choice.get("prompt_token_ids").filter(|v| !v.is_null()) {
                    match tokens(ids, self.max_tokens) {
                        Some(ids) if self.prompt.is_empty() || self.prompt == ids => {
                            self.prompt = ids
                        }
                        _ => self.invalid = true,
                    }
                }
                if let Some(ids) = choice.get("token_ids").filter(|v| !v.is_null()) {
                    if let Some(ids) =
                        tokens(ids, self.max_tokens.saturating_sub(self.output.len()))
                    {
                        self.output.extend(ids);
                    } else {
                        self.invalid = true;
                    }
                }
                if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
                    if matches!(reason, "stop" | "length" | "tool_calls" | "function_call") {
                        self.finished = true;
                    } else {
                        self.invalid = true;
                    }
                }
            }
        }
    }

    fn feed(&mut self, bytes: &[u8]) {
        if self.invalid {
            return;
        }
        if !self.streaming {
            if self.event.len().saturating_add(bytes.len()) > self.limit {
                self.invalid = true;
                self.event.clear();
            } else {
                self.event.extend_from_slice(bytes);
            }
            return;
        }
        for &byte in bytes {
            if byte != b'\n' {
                if self.line.len() >= self.limit {
                    self.invalid = true;
                    self.line.clear();
                    self.event.clear();
                    return;
                }
                self.line.push(byte);
                continue;
            }
            if self.line.last() == Some(&b'\r') {
                self.line.pop();
            }
            let line = std::mem::take(&mut self.line);
            if line.is_empty() {
                if self.event == b"[DONE]" {
                    self.done = true;
                } else if !self.event.is_empty() {
                    match serde_json::from_slice::<Value>(&self.event) {
                        Ok(value) => self.record(&value),
                        Err(_) => self.invalid = true,
                    }
                }
                self.event.clear();
            } else if let Some(mut data) = line.strip_prefix(b"data:") {
                if data.first() == Some(&b' ') {
                    data = &data[1..];
                }
                if self.event.len().saturating_add(data.len() + 1) > self.limit {
                    self.invalid = true;
                    self.event.clear();
                    return;
                }
                if !self.event.is_empty() {
                    self.event.push(b'\n');
                }
                self.event.extend_from_slice(data);
            }
        }
    }

    fn finish(mut self) {
        if !self.streaming && !self.invalid {
            match serde_json::from_slice::<Value>(&self.event) {
                Ok(value) => self.record(&value),
                Err(_) => self.invalid = true,
            }
        }
        if !self.invalid && self.finished && (!self.streaming || self.done) {
            if let Some(params) = self.params.take() {
                self.turn
                    .publish(params, &self.prompt, &self.output, &self.source);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};
    use http_body_util::BodyExt;

    fn worker() -> Arc<dyn Worker> {
        Arc::new(BasicWorker::new(
            "http://127.0.0.1:8200".into(),
            WorkerType::Decode,
        ))
    }
    fn params() -> Value {
        json!({"do_remote_decode":true,"do_remote_prefill":false,"remote_engine_id":"D-epoch-1","remote_request_id":"turn-1","remote_host":"advertised","remote_port":5600,"remote_block_ids":[[11,12]],"remote_num_tokens":3,"remote_blocks_expiry_time":123456.0,"transfer_mode":"pull"})
    }
    fn reply() -> Value {
        json!({"choices":[{"index":0,"prompt_token_ids":[1,2],"token_ids":[3,4],"text":"AB","finish_reason":"length"}],"kv_transfer_params":params()})
    }
    fn response(value: Value) -> Response {
        Response::builder()
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&value).unwrap()))
            .unwrap()
    }

    #[tokio::test]
    async fn native_completion_metadata_is_single_use_and_prefix_checked() {
        let cache = ConversationCache::new(BidirectionalKvConfig::default());
        let source = worker();
        let live = vec![source.clone()];
        let (turn, _, _) = cache.claim([1; 32], None, &live);
        let body = turn
            .unwrap()
            .observe(response(reply()), source.clone())
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        assert_eq!(body, serde_json::to_vec(&reply()).unwrap());
        let (next, remote, reason) = cache.claim([1; 32], Some(&[1, 2, 3, 99]), &live);
        assert_eq!(reason, "candidate");
        let remote = remote.unwrap();
        assert_eq!(remote["remote_host"], "advertised");
        assert_eq!(remote["remote_blocks_expiry_time"], 123456.0);
        drop(next);
        assert_eq!(cache.claim([1; 32], Some(&[1, 2, 3, 99]), &live).2, "miss");
    }

    #[tokio::test]
    async fn rewritten_history_and_replaced_engine_never_reuse() {
        for (key, current, replace, expected) in [
            ([2; 32], vec![1, 88, 3, 4], false, "prefix_mismatch"),
            ([3; 32], vec![1, 2, 3, 4], true, "source_removed"),
        ] {
            let cache = ConversationCache::new(BidirectionalKvConfig::default());
            let source = worker();
            let turn = cache
                .claim(key, None, std::slice::from_ref(&source))
                .0
                .unwrap();
            turn.observe(response(reply()), source.clone())
                .into_body()
                .collect()
                .await
                .unwrap();
            let live = if replace {
                vec![worker()]
            } else {
                vec![source]
            };
            let (_, remote, reason) = cache.claim(key, Some(&current), &live);
            assert!(remote.is_none());
            assert_eq!(reason, expected);
        }
    }

    #[test]
    fn concurrent_completion_and_cancel_cannot_publish_stale_metadata() {
        let cache = ConversationCache::new(BidirectionalKvConfig::default());
        let source = worker();
        let live = vec![source.clone()];
        let first = cache.claim([4; 32], None, &live).0.unwrap();
        assert_eq!(cache.claim([4; 32], None, &live).2, "concurrent_turn");
        first.publish(params(), &[1, 2], &[3, 4], &source);
        drop(first);
        let third = cache.claim([4; 32], Some(&[1, 2, 3]), &live);
        assert_eq!(third.2, "miss");
        drop(third);
        assert!(cache.state.lock().unwrap().slots.is_empty());
    }

    #[test]
    fn cache_expiry_does_not_compare_remote_monotonic_clocks() {
        let cache = ConversationCache::new(BidirectionalKvConfig::default());
        let source = worker();
        let live = vec![source.clone()];
        let turn = cache.claim([5; 32], None, &live).0.unwrap();
        turn.publish(params(), &[1, 2], &[3, 4], &source);
        drop(turn);
        cache
            .state
            .lock()
            .unwrap()
            .slots
            .get_mut(&[5; 32])
            .unwrap()
            .entry
            .as_mut()
            .unwrap()
            .expires = Instant::now();
        let (_, remote, reason) = cache.claim([5; 32], Some(&[1, 2, 3]), &live);
        assert!(remote.is_none());
        assert_eq!(reason, "expired");
    }

    #[tokio::test]
    async fn sse_split_at_every_byte_is_forwarded_unchanged() {
        let cache = ConversationCache::new(BidirectionalKvConfig::default());
        let source = worker();
        let live = vec![source.clone()];
        let first = json!({"prompt_token_ids":[1,2],"choices":[{"index":0,"delta":{"content":"中"},"token_ids":[3],"finish_reason":null}]});
        let last = json!({"choices":[{"index":0,"delta":{},"token_ids":[4],"finish_reason":"length"}],"kv_transfer_params":params()});
        let raw = format!("data: {first}\r\n\r\ndata: {last}\n\ndata: [DONE]\n\n").into_bytes();
        let chunks: Vec<_> = raw
            .iter()
            .map(|b| Ok::<_, std::io::Error>(bytes::Bytes::copy_from_slice(&[*b])))
            .collect();
        let response = Response::builder()
            .header("content-type", "text/event-stream")
            .body(Body::from_stream(futures_util::stream::iter(chunks)))
            .unwrap();
        let turn = cache.claim([6; 32], None, &live).0.unwrap();
        assert_eq!(
            turn.observe(response, source)
                .into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes(),
            raw
        );
        assert_eq!(
            cache.claim([6; 32], Some(&[1, 2, 3, 9]), &live).2,
            "candidate"
        );
    }

    #[tokio::test]
    async fn truncated_stream_and_client_drop_never_publish() {
        for drop_body in [false, true] {
            let cache = ConversationCache::new(BidirectionalKvConfig::default());
            let source = worker();
            let live = vec![source.clone()];
            let raw = format!("data: {}\n\n", reply());
            let response = Response::builder()
                .header("content-type", "text/event-stream")
                .body(Body::from(raw))
                .unwrap();
            let response = cache
                .claim([7; 32], None, &live)
                .0
                .unwrap()
                .observe(response, source);
            if drop_body {
                drop(response);
            } else {
                response.into_body().collect().await.unwrap();
            }
            assert!(cache.state.lock().unwrap().slots.is_empty());
        }
    }

    #[test]
    fn bounded_state_and_metadata_decline_without_affecting_requests() {
        let config = BidirectionalKvConfig {
            max_entries: 1,
            max_bytes: 64,
            max_entry_bytes: 64,
            ..BidirectionalKvConfig::default()
        };
        let cache = ConversationCache::new(config);
        let source = worker();
        let live = vec![source.clone()];
        let first = cache.claim([8; 32], None, &live).0.unwrap();
        assert_eq!(cache.claim([9; 32], None, &live).2, "capacity");
        first.publish(params(), &[1, 2], &[3, 4], &source);
        drop(first);
        assert!(cache.state.lock().unwrap().slots.is_empty());
    }

    #[test]
    fn credential_model_and_endpoint_scopes_do_not_alias() {
        let mut a = HeaderMap::new();
        a.insert("authorization", "Bearer one".parse().unwrap());
        let mut b = HeaderMap::new();
        b.insert("authorization", "Bearer two".parse().unwrap());
        let req = json!({"model":"m"});
        let key = request_key(&req, Some(&a), "c", "/v1/completions");
        assert_ne!(key, request_key(&req, Some(&b), "c", "/v1/completions"));
        assert_ne!(
            key,
            request_key(&json!({"model":"other"}), Some(&a), "c", "/v1/completions")
        );
        assert_ne!(
            key,
            request_key(&req, Some(&a), "c", "/v1/chat/completions")
        );
    }

    #[test]
    fn native_side_channel_host_is_preserved_and_missing_host_uses_http() {
        let cache = ConversationCache::new(BidirectionalKvConfig::default());
        let source = worker();
        let live = vec![source.clone()];
        for (key, native, expected) in [
            ([10; 32], Some("10.8.0.2"), "10.8.0.2"),
            ([11; 32], None, "127.0.0.1"),
        ] {
            let mut value = params();
            if let Some(host) = native {
                value["remote_host"] = json!(host);
            } else {
                value.as_object_mut().unwrap().remove("remote_host");
            }
            let turn = cache.claim(key, None, &live).0.unwrap();
            turn.publish(value, &[1, 2], &[3, 4], &source);
            drop(turn);
            let (_, remote, reason) = cache.claim(key, Some(&[1, 2, 3, 4]), &live);
            assert_eq!(reason, "candidate");
            assert_eq!(remote.unwrap()["remote_host"], expected);
        }
    }
}
