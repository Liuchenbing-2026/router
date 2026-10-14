//! Benchmarks for the P2 gRPC frontend render cache.
//!
//! Uses the same prompts and the same TinyLlama tokenizer as
//! `benches/tokenizer_cache_benchmark.rs` (#270), so the encode work that the
//! L0 encoding cache removes is identical and the costs are comparable. The P2
//! cache sits above L0: it removes the chat-template **render**, while every
//! encode still goes through L0.
//!
//! Groups:
//!   - `p2_cache/key`            digesting the whole wire request (per request)
//!   - `p2_cache/uncached_encode` the raw encode every request still pays (L0's job)
//!   - `p2_cache/hit`            key + LRU lookup + rendered-prompt clone
//!   - `p2_cache/miss`           key + miss + insert (fresh cache)
//!
//! Run with: `cargo bench --bench p2_request_cache_bench`

use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput,
};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use vllm_router_rs::backend::tokenize_cache::{
    cache_key, CachedTokenize, TokenizeCache, TokenizeCacheConfig,
};
use vllm_router_rs::protocols::spec::ChatCompletionRequest;
use vllm_router_rs::tokenizer::{huggingface::HuggingFaceTokenizer, traits::*};
use vllm_text::Prompt;

fn tokenizer_path() -> PathBuf {
    if let Ok(p) = std::env::var("P2_TOKENIZER_JSON") {
        return PathBuf::from(p);
    }
    let local = Path::new(".tokenizer_cache/tinyllama_tokenizer.json");
    if local.exists() {
        return local.to_path_buf();
    }
    PathBuf::from("/data1/lcb/codex/vllm_router/.tokenizer_cache/tinyllama_tokenizer.json")
}

fn load_tokenizer() -> Arc<dyn Tokenizer> {
    Arc::new(
        HuggingFaceTokenizer::from_file(tokenizer_path().to_str().unwrap())
            .expect("Failed to load tokenizer"),
    )
}

const SHORT_PROMPT: &str = "What is the capital of France?";
const MEDIUM_PROMPT: &str = "Write a detailed explanation of quantum computing, including its principles, current applications, and future potential. Be sure to cover both the theoretical foundations and practical implementations.";
const LONG_PROMPT: &str = "You are an expert software engineer. Review the following code and provide detailed feedback on performance optimizations, potential bugs, and architectural improvements. Consider scalability, maintainability, and best practices. The code implements a distributed caching system with the following requirements: 1) High availability across multiple regions, 2) Sub-millisecond latency for cache hits, 3) Automatic failover and recovery, 4) Support for both LRU and LFU eviction policies, 5) Real-time monitoring and alerting. Please analyze each component thoroughly and suggest concrete improvements with code examples where appropriate.";

fn generate_system_prompt(size: usize) -> String {
    let domains = [
        "mathematics",
        "physics",
        "chemistry",
        "biology",
        "computer science",
        "engineering",
        "medicine",
        "law",
        "economics",
        "philosophy",
    ];
    let mut prompt = String::from("You are a helpful AI assistant with expertise in ");
    let mut i = 0;
    while prompt.len() < size {
        prompt.push_str(domains[i % domains.len()]);
        prompt.push_str(", ");
        i += 1;
    }
    prompt
}

fn prompts() -> Vec<(&'static str, String)> {
    vec![
        ("short_30B", SHORT_PROMPT.to_string()),
        ("medium_201B", MEDIUM_PROMPT.to_string()),
        ("long_638B", LONG_PROMPT.to_string()),
        ("system_4KB", generate_system_prompt(4000)),
        ("system_16KB", generate_system_prompt(16000)),
    ]
}

/// The gRPC path sees a full chat request, not a bare string: the cache key
/// covers the wire request, so the bench must build one.
fn request_for(text: &str) -> ChatCompletionRequest {
    serde_json::from_value(serde_json::json!({
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "messages": [{ "role": "user", "content": text }],
        "max_tokens": 8,
        "add_generation_prompt": true
    }))
    .expect("valid chat request")
}

fn value_for(text: &str) -> CachedTokenize {
    CachedTokenize {
        prompt: Prompt::Text(text.to_string()),
        template_ms: 0.0,
    }
}

fn bench_p2(c: &mut Criterion) {
    let inner = load_tokenizer();
    let cfg = TokenizeCacheConfig::default();

    // Per-request key cost: paid on every request, hit or miss.
    let mut g = c.benchmark_group("p2_cache/key");
    for (name, prompt) in prompts() {
        let req = request_for(&prompt);
        g.throughput(Throughput::Bytes(prompt.len() as u64));
        g.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter(|| black_box(cache_key(black_box(&req)).unwrap()))
        });
    }
    g.finish();

    // Reference: the raw encode a hit avoids (same work #270's bench measures).
    let mut g = c.benchmark_group("p2_cache/uncached_encode");
    for (name, prompt) in prompts() {
        g.throughput(Throughput::Bytes(prompt.len() as u64));
        g.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter_with_large_drop(|| inner.encode(black_box(&prompt)).unwrap())
        });
    }
    g.finish();

    // Hit: key + lookup + rendered-prompt clone.
    let mut g = c.benchmark_group("p2_cache/hit");
    for (name, prompt) in prompts() {
        let req = request_for(&prompt);
        let key = cache_key(&req).unwrap();
        let cache = TokenizeCache::new(cfg).unwrap();
        cache.put(key, value_for(&prompt));
        g.throughput(Throughput::Bytes(prompt.len() as u64));
        g.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter(|| {
                let k = cache_key(black_box(&req)).unwrap();
                black_box(cache.get(black_box(&k)))
            })
        });
    }
    g.finish();

    // Miss: key + lookup + insert into a fresh cache.
    let mut g = c.benchmark_group("p2_cache/miss");
    for (name, prompt) in prompts() {
        let req = request_for(&prompt);
        g.throughput(Throughput::Bytes(prompt.len() as u64));
        g.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter_batched(
                || TokenizeCache::new(cfg).unwrap(),
                |cache| {
                    let k = cache_key(black_box(&req)).unwrap();
                    cache.put(k, value_for(black_box(&prompt)));
                },
                BatchSize::SmallInput,
            )
        });
    }
    g.finish();
}

criterion_group!(benches, bench_p2);
criterion_main!(benches);
