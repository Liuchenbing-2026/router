//! Bounded exact-result **render** cache for the gRPC frontend.
//!
//! The gRPC worker path renders a chat request and encodes the rendered prompt
//! on every request, in process, before the policy picks a worker. Agent
//! workloads resend the same conversation and vary only the tail, so the
//! render step repeats far more often than the router's worker-side KV reuse
//! can absorb.
//!
//! This cache memoizes the **render** for an exact request and nothing else. It
//! is deliberately not [`crate::tokenizer::CachedTokenizer`] (aka
//! `super::encode_cache::FrontendCache`): that type memoizes `text -> ids` for
//! the encode step. Keeping the two apart means this layer's hits never hide an
//! encode from L0, so L0's hit/miss accounting stays faithful while render time
//! is still skipped on a repeat. This cache has the same bounds, the same
//! oversize rule and its own metrics as L0.
//!
//! # Semantics
//!
//! - A hit returns an owned clone of the rendered prompt and its render clock,
//!   so callers can keep attributing time without pretending a cached request
//!   did new work.
//! - Only successful results are stored. Errors leave the cache untouched.
//! - Rendering runs outside the cache lock, so two concurrent misses on the
//!   same key both render and the later insert wins.
//! - The cache is keyed on the request digest, which covers every field the
//!   request lowering reads. Fields that do not affect tokenization (sampling
//!   parameters, for example) are hashed too: including them costs hit rate and
//!   never costs correctness, because equal keys imply equal requests.
//!
//! # Bounds and eviction
//!
//! Bounded by entry count and an estimated retained-byte budget. Results whose
//! estimate exceeds `max_entry_bytes` are returned to the caller but not stored.
//! Eviction is least-recently-used.
//!
//! # Metrics
//!
//! Hits, misses, evictions and oversize skips are counted under
//! `vllm_router_tokenize_cache_*_total`; `vllm_router_tokenize_cache_entries`
//! and `vllm_router_tokenize_cache_bytes` are gauges for this cache's
//! occupancy.

use crate::metrics::TokenizerMetrics;
use crate::protocols::spec::ChatCompletionRequest;
use anyhow::{bail, Result};
use lru::LruCache;
use parking_lot::Mutex;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use vllm_text::Prompt;

/// Digest length of [`cache_key`].
pub const KEY_LEN: usize = 32;

/// Fixed per-entry bookkeeping charged against the byte budget.
const ENTRY_OVERHEAD_BYTES: usize = 64;

/// Bytes charged per cached token ID.
const BYTES_PER_TOKEN_ID: usize = std::mem::size_of::<u32>();

/// Budgets for a [`TokenizeCache`]. Mirrors
/// [`crate::tokenizer::TokenizerCacheConfig`] so operators tune one shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenizeCacheConfig {
    /// Maximum number of entries retained. Must be at least 1.
    pub max_entries: usize,
    /// Maximum estimated bytes retained across all entries. Must be at least 1.
    pub max_bytes: usize,
    /// Results whose estimate exceeds this are not cached. Must be at least 1
    /// and at most `max_bytes`.
    pub max_entry_bytes: usize,
}

impl Default for TokenizeCacheConfig {
    fn default() -> Self {
        Self {
            // A turn of an agent session is one entry; 10k covers a large fan-out
            // over a handful of long-lived sessions.
            max_entries: 10_000,
            max_bytes: 64 * 1024 * 1024,
            max_entry_bytes: 1024 * 1024,
        }
    }
}

impl TokenizeCacheConfig {
    /// Check that every budget is usable.
    pub fn validate(&self) -> Result<()> {
        if self.max_entries == 0 {
            bail!("tokenize cache max_entries must be at least 1");
        }
        if self.max_bytes == 0 {
            bail!("tokenize cache max_bytes must be at least 1");
        }
        if self.max_entry_bytes == 0 {
            bail!("tokenize cache max_entry_bytes must be at least 1");
        }
        if self.max_entry_bytes > self.max_bytes {
            bail!("tokenize cache max_entry_bytes must not exceed max_bytes");
        }
        Ok(())
    }
}

/// One cached chat-template render result.
///
/// Only the rendered [`Prompt`] and its render time are stored. Encoding is
/// left to the shared L0 encoding cache (`crate::tokenizer::CachedTokenizer` /
/// `super::encode_cache::FrontendCache`) so this layer's hits never hide an
/// encode from L0's hit/miss accounting.
#[derive(Debug, Clone)]
pub struct CachedTokenize {
    /// The rendered prompt, cloned to every caller that hits this entry.
    pub prompt: Prompt,
    /// Chat-template time of the render that produced this prompt, in ms.
    pub template_ms: f64,
}

/// A point-in-time copy of the cache counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TokenizeCacheStats {
    pub entries: usize,
    pub bytes: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub oversized: u64,
}

/// Bounded exact-result cache for the gRPC render+encode step.
pub struct TokenizeCache {
    config: TokenizeCacheConfig,
    entries: Mutex<LruCache<[u8; KEY_LEN], Entry>>,
    bytes: AtomicUsize,
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    oversized: AtomicU64,
}

/// A stored result plus the bytes it is charged against the budget.
struct Entry {
    value: CachedTokenize,
    bytes: usize,
}

impl std::fmt::Debug for TokenizeCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenizeCache")
            .field("config", &self.config)
            .field("stats", &self.stats())
            .finish()
    }
}

impl TokenizeCache {
    /// Build a cache from validated budgets.
    pub fn new(config: TokenizeCacheConfig) -> Result<Self> {
        config.validate()?;
        let capacity =
            NonZeroUsize::new(config.max_entries).expect("validate() rejects a zero entry budget");
        Ok(Self {
            config,
            entries: Mutex::new(LruCache::sparse(capacity)),
            bytes: AtomicUsize::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            oversized: AtomicU64::new(0),
        })
    }

    /// Look up a request digest, cloning the stored result on a hit.
    pub fn get(&self, key: &[u8; KEY_LEN]) -> Option<CachedTokenize> {
        let value = self
            .entries
            .lock()
            .get(key)
            .map(|entry| entry.value.clone());
        match &value {
            Some(_) => {
                self.hits.fetch_add(1, Ordering::Relaxed);
                TokenizerMetrics::record_tokenize_cache_hit();
            }
            None => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                TokenizerMetrics::record_tokenize_cache_miss();
            }
        }
        value
    }

    /// Store a result. Entries larger than `max_entry_bytes` are counted and
    /// dropped, so one very long conversation cannot flush the working set.
    pub fn put(&self, key: [u8; KEY_LEN], value: CachedTokenize) {
        let bytes = estimate_entry_bytes(&value);
        if bytes > self.config.max_entry_bytes {
            self.oversized.fetch_add(1, Ordering::Relaxed);
            TokenizerMetrics::record_tokenize_cache_oversized();
            return;
        }

        let mut evicted: u64 = 0;
        {
            let mut entries = self.entries.lock();
            if let Some((_, old)) = entries.push(key, Entry { value, bytes }) {
                self.bytes.fetch_sub(old.bytes, Ordering::Relaxed);
            }
            self.bytes.fetch_add(bytes, Ordering::Relaxed);
            // `max_entry_bytes <= max_bytes` is validated, so this never evicts
            // the entry just pushed.
            while self.bytes.load(Ordering::Relaxed) > self.config.max_bytes {
                match entries.pop_lru() {
                    Some((_, victim)) => {
                        self.bytes.fetch_sub(victim.bytes, Ordering::Relaxed);
                        evicted += 1;
                    }
                    None => break,
                }
            }
        }

        if evicted > 0 {
            self.evictions.fetch_add(evicted, Ordering::Relaxed);
            for _ in 0..evicted {
                TokenizerMetrics::record_tokenize_cache_eviction();
            }
        }
        self.publish_occupancy();
    }

    /// Drop every entry. Retained bytes are released outside the lock.
    pub fn clear(&self) {
        let removed = {
            let mut entries = self.entries.lock();
            let capacity = NonZeroUsize::new(self.config.max_entries)
                .expect("validate() rejects a zero entry budget");
            let removed = std::mem::replace(&mut *entries, LruCache::sparse(capacity));
            self.bytes.store(0, Ordering::Relaxed);
            removed
        };
        drop(removed);
        self.publish_occupancy();
    }

    /// Copy the counters.
    pub fn stats(&self) -> TokenizeCacheStats {
        TokenizeCacheStats {
            entries: self.entries.lock().len(),
            bytes: self.bytes.load(Ordering::Relaxed),
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            oversized: self.oversized.load(Ordering::Relaxed),
        }
    }

    /// Publish this cache's occupancy. The gRPC path keeps one cache per
    /// process, so the gauges are a set rather than an aggregate.
    fn publish_occupancy(&self) {
        let entries = self.entries.lock().len();
        TokenizerMetrics::record_tokenize_cache_occupancy(
            entries,
            self.bytes.load(Ordering::Relaxed),
        );
    }
}

/// Estimate the heap bytes an entry keeps alive: the rendered text (or the
/// already-tokenized ids) plus per-entry bookkeeping. A lower bound, matching
/// the tokenizer cache's rule.
fn estimate_entry_bytes(value: &CachedTokenize) -> usize {
    let payload = match &value.prompt {
        Prompt::Text(text) => text.len(),
        Prompt::TokenIds(ids) => ids.len() * BYTES_PER_TOKEN_ID,
    };
    ENTRY_OVERHEAD_BYTES + payload
}

/// Digest the fields that decide the token IDs of a chat request.
///
/// Hashing the whole wire request is the safe choice: every field the lowering
/// reads is covered, so two requests can share an entry only when they are
/// equal. `serde_json` is built with `preserve_order`, so `other` and the
/// message content maps serialise in arrival order; a field typed as a
/// `HashMap` can serialise in a different order for equal content, which only
/// turns a would-be hit into a miss.
pub fn cache_key(request: &ChatCompletionRequest) -> Result<[u8; KEY_LEN]> {
    let encoded = serialize(request)?;
    let mut hasher = Sha256::new();
    hasher.update((encoded.len() as u64).to_le_bytes());
    hasher.update(&encoded);
    Ok(hasher.finalize().into())
}

/// Serialize a request for hashing.
fn serialize<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(value)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(byte: u8) -> [u8; KEY_LEN] {
        [byte; KEY_LEN]
    }

    fn value(text_bytes: usize) -> CachedTokenize {
        CachedTokenize {
            prompt: Prompt::Text("x".repeat(text_bytes)),
            template_ms: 1.5,
        }
    }

    #[test]
    fn validates_budgets() {
        assert!(TokenizeCacheConfig::default().validate().is_ok());
        let zero_entries = TokenizeCacheConfig {
            max_entries: 0,
            ..Default::default()
        };
        assert!(zero_entries.validate().is_err());
        let entry_over_budget = TokenizeCacheConfig {
            max_bytes: 1024,
            max_entry_bytes: 2048,
            ..Default::default()
        };
        assert!(entry_over_budget.validate().is_err());
    }

    #[test]
    fn hit_returns_the_stored_result_and_counts() {
        let cache = TokenizeCache::new(TokenizeCacheConfig::default()).unwrap();
        cache.put(key(1), value(4));
        let got = cache.get(&key(1)).expect("hit");
        assert_eq!(got.prompt, Prompt::Text("xxxx".to_string()));
        assert_eq!(got.template_ms, 1.5);
        assert!(cache.get(&key(2)).is_none());
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.entries, 1);
    }

    #[test]
    fn oversized_entries_are_returned_but_not_stored() {
        let cache = TokenizeCache::new(TokenizeCacheConfig {
            max_entry_bytes: 512,
            ..Default::default()
        })
        .unwrap();
        cache.put(key(1), value(4096));
        assert!(cache.get(&key(1)).is_none());
        assert_eq!(cache.stats().oversized, 1);
        assert_eq!(cache.stats().entries, 0);
    }

    #[test]
    fn byte_budget_evicts_least_recently_used() {
        // Each stored entry is exactly this many bytes, so a budget of four
        // entries makes eviction deterministic.
        let entry_bytes = ENTRY_OVERHEAD_BYTES + 10;
        let cache = TokenizeCache::new(TokenizeCacheConfig {
            max_entries: 16,
            max_bytes: 4 * entry_bytes,
            max_entry_bytes: entry_bytes,
        })
        .unwrap();
        for byte in 0..6u8 {
            cache.put(key(byte), value(10));
        }
        let stats = cache.stats();
        assert!(stats.evictions >= 2, "evictions = {}", stats.evictions);
        assert!(stats.bytes <= 4 * entry_bytes);
        assert!(cache.get(&key(0)).is_none(), "the oldest entry survived");
        assert!(cache.get(&key(5)).is_some(), "the newest entry was evicted");
    }

    #[test]
    fn clear_releases_every_entry() {
        let cache = TokenizeCache::new(TokenizeCacheConfig::default()).unwrap();
        cache.put(key(1), value(8));
        cache.clear();
        let stats = cache.stats();
        assert_eq!(stats.entries, 0);
        assert_eq!(stats.bytes, 0);
    }

    #[test]
    fn key_is_stable_for_equal_requests_and_differs_otherwise() {
        let request = |text: &str| {
            serde_json::from_value::<ChatCompletionRequest>(serde_json::json!({
                "messages": [{"role": "user", "content": text}]
            }))
            .expect("the fixture is a valid chat request")
        };
        assert_eq!(
            cache_key(&request("hello")).unwrap(),
            cache_key(&request("hello")).unwrap()
        );
        assert_ne!(
            cache_key(&request("hello")).unwrap(),
            cache_key(&request("world")).unwrap()
        );
    }
}
