//! #359: what an L0 hit costs before and after the `Arc<[u32]>` hand-off.
//!
//! Before, a hit cloned the cached `Vec<u32>` and preprocessing then converted
//! it to `Arc<[u32]>`: two copies of the whole prompt's token IDs on the hit
//! path. After, the cache stores `Arc<[u32]>` and a hit is a refcount bump that
//! the ids travel on straight into `PreparedChat`.
//!
//! Groups:
//!   - `l0_handoff/before`  the old `Arc::from(cached.clone())`
//!   - `l0_handoff/after`   the new `Arc::clone(&cached)`
//!
//! Run with `cargo bench --bench l0_arc_handoff`.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::sync::Arc;

fn sizes() -> [(&'static str, usize); 3] {
    [
        ("short_512", 512),
        ("turn_4k", 4 * 1024),
        ("long_32k", 32 * 1024),
    ]
}

fn bench_handoff(c: &mut Criterion) {
    let mut group = c.benchmark_group("l0_handoff");
    for (name, tokens) in sizes() {
        let cached: Vec<u32> = (0..tokens as u32).collect();
        let cached_arc: Arc<[u32]> = Arc::from(cached.clone());
        group.throughput(Throughput::Elements(tokens as u64));
        group.bench_with_input(BenchmarkId::new("before", name), &cached, |b, cached| {
            b.iter(|| {
                let cloned = cached.clone();
                black_box(Arc::<[u32]>::from(cloned))
            });
        });
        group.bench_with_input(BenchmarkId::new("after", name), &cached_arc, |b, cached| {
            b.iter(|| black_box(Arc::clone(cached)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_handoff);
criterion_main!(benches);
