//! Portable reconciliation snapshots for restarting or newly attached Routers.
//!
//! A checksum detects accidental corruption. It does not authenticate the
//! producer and cannot make request metadata trusted. Only the ingestion owner
//! may pass a snapshot to the runtime importer.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{HashMode, ResidencyOwner, RoutableTarget, StorageTier};

pub const PORTABLE_SNAPSHOT_FORMAT_VERSION: u32 = 2;
const MAX_IDENTIFIER_LEN: usize = 1024;
pub(crate) const SNAPSHOT_CLOCK_SKEW_MS: u64 = 1_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PortableHashMode {
    PythonHashSeed,
    Sha256,
    XxhashCbor,
}

impl PortableHashMode {
    pub(crate) fn matches(self, mode: HashMode) -> bool {
        matches!(
            (self, mode),
            (Self::PythonHashSeed, HashMode::PythonHashSeed)
                | (Self::Sha256, HashMode::Sha256)
                | (Self::XxhashCbor, HashMode::XxhashCbor)
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PortableCacheIdentity {
    pub model: String,
    pub tokenizer: String,
    pub hash_mode: PortableHashMode,
    pub block_size: u32,
}

impl PortableCacheIdentity {
    pub fn new(
        model: impl Into<String>,
        tokenizer: impl Into<String>,
        hash_mode: PortableHashMode,
        block_size: u32,
    ) -> Self {
        Self {
            model: model.into(),
            tokenizer: tokenizer.into(),
            hash_mode,
            block_size,
        }
    }
}

/// A control-plane ownership token carried by a portable snapshot.
///
/// The snapshot only declares the token. The runtime importer must also receive
/// the same token from its trusted control-plane integration before accepting
/// the snapshot. This type is not a signature or an authentication credential.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PortableSnapshotFence {
    pub owner_id: String,
    pub epoch: u64,
}

impl PortableSnapshotFence {
    pub fn new(owner_id: impl Into<String>, epoch: u64) -> Self {
        Self {
            owner_id: owner_id.into(),
            epoch,
        }
    }
}

/// Durable control-plane authority for portable-snapshot ownership.
///
/// Implementations must atomically reject an epoch rollback, reject an owner
/// change within one epoch, and advance the authoritative fence. The runtime
/// calls this only after the complete snapshot and its staged state have
/// passed validation, outside the live KV state lock. Implementations must
/// treat replay of the exact same fence as success because a concurrent live
/// update can require the caller to retry the snapshot import.
///
/// This adapter contract is not leader election or snapshot authentication.
/// A production implementation should persist the fence in the same authority
/// that authenticates snapshot producers and allocates epochs.
pub trait PortableSnapshotFenceAuthority: Send + Sync {
    fn admit_and_advance(&self, fence: &PortableSnapshotFence)
        -> Result<(), PortableSnapshotError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PortableStorageTier {
    Device,
    HostPinned,
    Disk,
    External,
}

impl From<PortableStorageTier> for StorageTier {
    fn from(value: PortableStorageTier) -> Self {
        match value {
            PortableStorageTier::Device => StorageTier::Device,
            PortableStorageTier::HostPinned => StorageTier::HostPinned,
            PortableStorageTier::Disk => StorageTier::Disk,
            PortableStorageTier::External => StorageTier::External,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum PortableResidencyOwner {
    Worker {
        instance_id: String,
        dp_rank: u32,
        incarnation: u64,
    },
    CacheOwner {
        pool_id: String,
        source_instance_id: String,
        dp_rank: u32,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PortableBlock {
    /// Opaque engine sequence/block identity, in the runtime's `b:`/`i:` form.
    pub block_hash: String,
    /// Router-local hash used for prefix matching.
    pub local_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PortablePrefix {
    /// A block already present in this source snapshot, or `None` for a prefix
    /// rooted at the empty sequence.
    pub parent_block_hash: Option<String>,
    pub blocks: Vec<PortableBlock>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PortableResidency {
    pub owner: PortableResidencyOwner,
    pub tier: PortableStorageTier,
    pub prefixes: Vec<PortablePrefix>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PortableSourceSnapshot {
    pub instance_id: String,
    pub dp_rank: u32,
    pub incarnation: u64,
    pub sequence: u64,
    /// Unix epoch milliseconds at which the complete source view was observed.
    pub observed_at_unix_ms: u64,
    pub residencies: Vec<PortableResidency>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PortableKvSnapshot {
    pub format_version: u32,
    pub snapshot_id: String,
    /// A false value is an incomplete transfer and is never importable.
    pub complete: bool,
    pub cache_identity: PortableCacheIdentity,
    pub created_at_unix_ms: u64,
    pub sources: Vec<PortableSourceSnapshot>,
    /// Fencing token asserted by the trusted control-plane integration.
    pub fence: PortableSnapshotFence,
    /// SHA-256 over the JSON form with this field cleared.
    pub checksum_sha256: String,
}

impl PortableKvSnapshot {
    pub fn seal(&mut self) -> Result<(), PortableSnapshotError> {
        self.checksum_sha256 = self.compute_checksum()?;
        Ok(())
    }

    pub fn compute_checksum(&self) -> Result<String, PortableSnapshotError> {
        let mut canonical = self.clone();
        canonical.checksum_sha256.clear();
        let bytes = serde_json::to_vec(&canonical)
            .map_err(|_| PortableSnapshotError::Invalid("snapshot is not serializable"))?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PortableSnapshotError {
    #[error("invalid portable snapshot: {0}")]
    Invalid(&'static str),
    #[error("snapshot cache identity does not match the trusted expectation")]
    IdentityMismatch,
    #[error("snapshot checksum does not match its payload")]
    ChecksumMismatch,
    #[error("snapshot is incomplete")]
    Incomplete,
    #[error("snapshot source set does not match the discovered targets")]
    SourceSetMismatch,
    #[error("snapshot is older than the applied snapshot")]
    Rollback,
    #[error("snapshot incarnation is older than the applied incarnation")]
    ObsoleteIncarnation,
    #[error("snapshot id was reused with different content")]
    SnapshotIdConflict,
    #[error("snapshot source content conflicts with the applied sequence")]
    SnapshotConflict,
    #[error("snapshot fence does not match the trusted control-plane expectation")]
    FenceMismatch,
    #[error("snapshot fencing epoch is older than the applied epoch")]
    StaleEpoch,
    #[error("one fencing epoch cannot change owner")]
    EpochOwnerConflict,
    #[error("KV state changed while the snapshot fence was being committed; retry the same fence")]
    ConcurrentSnapshotMutation,
    #[error("snapshot exceeds the configured KV block budget")]
    BlockBudgetExceeded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortableSnapshotImportStatus {
    Applied,
    AlreadyApplied,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortableSnapshotImport {
    pub snapshot_id: String,
    pub status: PortableSnapshotImportStatus,
    pub sources: usize,
    pub blocks: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct ValidatedPortableSnapshot {
    pub snapshot_id: String,
    pub created_at_unix_ms: u64,
    pub checksum_sha256: String,
    pub fence: PortableSnapshotFence,
    pub sources: Vec<ValidatedPortableSource>,
    pub block_count: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct ValidatedPortableSource {
    pub target: RoutableTarget,
    pub incarnation: u64,
    pub sequence: u64,
    pub observed_at_unix_ms: u64,
    pub content_sha256: String,
    pub residencies: Vec<ValidatedPortableResidency>,
}

#[derive(Debug, Clone)]
pub(crate) struct ValidatedPortableResidency {
    pub owner: ResidencyOwner,
    pub tier: StorageTier,
    pub prefixes: Vec<ValidatedPortablePrefix>,
}

#[derive(Debug, Clone)]
pub(crate) struct ValidatedPortablePrefix {
    pub parent_block_hash: Option<Arc<str>>,
    pub blocks: Vec<(Arc<str>, Arc<str>)>,
}

/// Validate bounds, identity, integrity and parent-chain completeness before
/// any Router state is changed.
pub(crate) fn validate_portable_snapshot(
    snapshot: &PortableKvSnapshot,
    expected_identity: &PortableCacheIdentity,
    expected_fence: &PortableSnapshotFence,
    max_sources: usize,
    max_blocks: usize,
    now_unix_ms: u64,
) -> Result<ValidatedPortableSnapshot, PortableSnapshotError> {
    if snapshot.format_version != PORTABLE_SNAPSHOT_FORMAT_VERSION {
        return Err(PortableSnapshotError::Invalid(
            "unsupported snapshot format version",
        ));
    }
    validate_text(&snapshot.snapshot_id)?;
    validate_identity(&snapshot.cache_identity)?;
    validate_identity(expected_identity)?;
    validate_fence(&snapshot.fence)?;
    validate_fence(expected_fence)?;
    if snapshot.cache_identity != *expected_identity {
        return Err(PortableSnapshotError::IdentityMismatch);
    }
    if snapshot.fence != *expected_fence {
        return Err(PortableSnapshotError::FenceMismatch);
    }
    if !snapshot.complete {
        return Err(PortableSnapshotError::Incomplete);
    }
    if snapshot.checksum_sha256.len() != 64
        || !snapshot
            .checksum_sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
        || snapshot.checksum_sha256 != snapshot.compute_checksum()?
    {
        return Err(PortableSnapshotError::ChecksumMismatch);
    }
    if snapshot.created_at_unix_ms > now_unix_ms.saturating_add(SNAPSHOT_CLOCK_SKEW_MS) {
        return Err(PortableSnapshotError::Invalid(
            "snapshot creation time is in the future",
        ));
    }
    if max_sources == 0 || max_blocks == 0 {
        return Err(PortableSnapshotError::Invalid(
            "snapshot capacities must be positive",
        ));
    }
    if snapshot.sources.len() > max_sources {
        return Err(PortableSnapshotError::Invalid(
            "snapshot source budget exceeded",
        ));
    }

    let mut seen_targets = HashSet::new();
    let mut sources = Vec::with_capacity(snapshot.sources.len());
    let mut block_count = 0usize;
    for source in &snapshot.sources {
        validate_text(&source.instance_id)?;
        if !seen_targets.insert((source.instance_id.as_str(), source.dp_rank)) {
            return Err(PortableSnapshotError::Invalid(
                "duplicate snapshot source target",
            ));
        }
        if source.observed_at_unix_ms > now_unix_ms.saturating_add(SNAPSHOT_CLOCK_SKEW_MS) {
            return Err(PortableSnapshotError::Invalid(
                "snapshot observation time is in the future",
            ));
        }
        if source.observed_at_unix_ms
            > snapshot
                .created_at_unix_ms
                .saturating_add(SNAPSHOT_CLOCK_SKEW_MS)
        {
            return Err(PortableSnapshotError::Invalid(
                "source observation is newer than the snapshot",
            ));
        }
        let target = RoutableTarget {
            instance_id: Arc::from(source.instance_id.as_str()),
            dp_rank: source.dp_rank,
        };
        let mut seen_owner_tiers = HashSet::new();
        let mut residencies = Vec::with_capacity(source.residencies.len());
        let mut source_blocks = HashMap::<Arc<str>, Arc<str>>::new();
        for residency in &source.residencies {
            let owner = validate_owner(
                &residency.owner,
                &source.instance_id,
                source.dp_rank,
                source.incarnation,
            )?;
            let owner_tier = (owner.clone(), residency.tier);
            if !seen_owner_tiers.insert(owner_tier) {
                return Err(PortableSnapshotError::Invalid(
                    "duplicate snapshot owner and tier",
                ));
            }
            let mut seen_blocks = HashSet::new();
            let mut prefixes = Vec::with_capacity(residency.prefixes.len());
            for prefix in &residency.prefixes {
                if prefix.blocks.is_empty() {
                    return Err(PortableSnapshotError::Invalid(
                        "snapshot prefix has no blocks",
                    ));
                }
                if let Some(parent) = &prefix.parent_block_hash {
                    validate_text(parent)?;
                }
                let mut blocks = Vec::with_capacity(prefix.blocks.len());
                for block in &prefix.blocks {
                    validate_text(&block.block_hash)?;
                    validate_text(&block.local_hash)?;
                    if !seen_blocks.insert(block.block_hash.clone()) {
                        return Err(PortableSnapshotError::Invalid(
                            "duplicate block claim in snapshot residency",
                        ));
                    }
                    let block_hash: Arc<str> = Arc::from(block.block_hash.as_str());
                    let local_hash: Arc<str> = Arc::from(block.local_hash.as_str());
                    if let Some(previous) =
                        source_blocks.insert(block_hash.clone(), local_hash.clone())
                    {
                        if previous != local_hash {
                            return Err(PortableSnapshotError::Invalid(
                                "one block hash has inconsistent local hashes",
                            ));
                        }
                    }
                    blocks.push((block_hash, local_hash));
                    block_count = block_count
                        .checked_add(1)
                        .ok_or(PortableSnapshotError::BlockBudgetExceeded)?;
                    if block_count > max_blocks {
                        return Err(PortableSnapshotError::BlockBudgetExceeded);
                    }
                }
                prefixes.push(ValidatedPortablePrefix {
                    parent_block_hash: prefix.parent_block_hash.as_deref().map(Arc::from),
                    blocks,
                });
            }
            residencies.push(ValidatedPortableResidency {
                owner,
                tier: residency.tier.into(),
                prefixes,
            });
        }
        validate_prefix_order(&source.instance_id, source.dp_rank, &residencies)?;
        sources.push(ValidatedPortableSource {
            target,
            incarnation: source.incarnation,
            sequence: source.sequence,
            observed_at_unix_ms: source.observed_at_unix_ms,
            content_sha256: source_content_sha256(source)?,
            residencies,
        });
    }
    Ok(ValidatedPortableSnapshot {
        snapshot_id: snapshot.snapshot_id.clone(),
        created_at_unix_ms: snapshot.created_at_unix_ms,
        checksum_sha256: snapshot.checksum_sha256.clone(),
        fence: snapshot.fence.clone(),
        sources,
        block_count,
    })
}

fn validate_identity(identity: &PortableCacheIdentity) -> Result<(), PortableSnapshotError> {
    validate_text(&identity.model)?;
    validate_text(&identity.tokenizer)?;
    if identity.block_size == 0 {
        return Err(PortableSnapshotError::Invalid(
            "snapshot cache block size must be positive",
        ));
    }
    Ok(())
}

fn validate_fence(fence: &PortableSnapshotFence) -> Result<(), PortableSnapshotError> {
    validate_text(&fence.owner_id)
}

fn validate_text(value: &str) -> Result<(), PortableSnapshotError> {
    if value.is_empty() {
        return Err(PortableSnapshotError::Invalid(
            "snapshot identity must not be empty",
        ));
    }
    if value.len() > MAX_IDENTIFIER_LEN {
        return Err(PortableSnapshotError::Invalid(
            "snapshot identity exceeds the length limit",
        ));
    }
    Ok(())
}

/// Hash the immutable state represented by one source at an event-log position.
/// Snapshot metadata and observation time are intentionally excluded: another
/// producer may observe the same position later, but it must not describe a
/// different topology for the same incarnation and sequence.
fn source_content_sha256(source: &PortableSourceSnapshot) -> Result<String, PortableSnapshotError> {
    let mut residencies: Vec<_> = source
        .residencies
        .iter()
        .cloned()
        .map(|mut residency| {
            residency.prefixes.sort_by_key(canonical_prefix_key);
            serde_json::to_vec(&residency)
                .map_err(|_| PortableSnapshotError::Invalid("snapshot is not serializable"))
        })
        .collect::<Result<_, _>>()?;
    residencies.sort();

    let mut hasher = Sha256::new();
    hash_part(&mut hasher, source.instance_id.as_bytes())?;
    hasher.update(source.dp_rank.to_be_bytes());
    hasher.update(source.incarnation.to_be_bytes());
    hasher.update(source.sequence.to_be_bytes());
    hasher.update((residencies.len() as u64).to_be_bytes());
    for residency in residencies {
        hash_part(&mut hasher, &residency)?;
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn canonical_prefix_key(prefix: &PortablePrefix) -> String {
    let parent = prefix.parent_block_hash.as_deref().unwrap_or("");
    let mut key = format!("{}:{parent}:{}:", parent.len(), prefix.blocks.len());
    for block in &prefix.blocks {
        key.push_str(&format!(
            "{}:{}:{}:{}:",
            block.block_hash.len(),
            block.block_hash,
            block.local_hash.len(),
            block.local_hash
        ));
    }
    key
}

fn hash_part(hasher: &mut Sha256, bytes: &[u8]) -> Result<(), PortableSnapshotError> {
    let len = u64::try_from(bytes.len())
        .map_err(|_| PortableSnapshotError::Invalid("snapshot field exceeds the hash range"))?;
    hasher.update(len.to_be_bytes());
    hasher.update(bytes);
    Ok(())
}

fn validate_owner(
    owner: &PortableResidencyOwner,
    instance_id: &str,
    dp_rank: u32,
    source_incarnation: u64,
) -> Result<ResidencyOwner, PortableSnapshotError> {
    match owner {
        PortableResidencyOwner::Worker {
            instance_id: owner_instance,
            dp_rank: owner_rank,
            incarnation,
        } => {
            validate_text(owner_instance)?;
            if owner_instance != instance_id || *owner_rank != dp_rank {
                return Err(PortableSnapshotError::Invalid(
                    "worker residency is outside its snapshot source",
                ));
            }
            if *incarnation != source_incarnation {
                return Err(PortableSnapshotError::Invalid(
                    "worker residency incarnation does not match its snapshot source",
                ));
            }
            Ok(ResidencyOwner::Worker {
                source: Arc::from(owner_instance.as_str()),
                dp_rank: *owner_rank,
                incarnation: *incarnation,
            })
        }
        PortableResidencyOwner::CacheOwner {
            pool_id,
            source_instance_id,
            dp_rank: owner_rank,
        } => {
            validate_text(pool_id)?;
            validate_text(source_instance_id)?;
            if source_instance_id != instance_id || *owner_rank != dp_rank {
                return Err(PortableSnapshotError::Invalid(
                    "shared residency is attributed outside its snapshot source",
                ));
            }
            Ok(ResidencyOwner::CacheOwner {
                pool_id: Arc::from(pool_id.as_str()),
                source: Arc::from(source_instance_id.as_str()),
                dp_rank: *owner_rank,
            })
        }
    }
}

/// Prefixes may refer to an earlier prefix in the same source. Reject dangling
/// parents and cycles so staging can never silently skip an unanchored suffix.
fn validate_prefix_order(
    instance_id: &str,
    dp_rank: u32,
    residencies: &[ValidatedPortableResidency],
) -> Result<(), PortableSnapshotError> {
    let all_hashes: HashSet<Arc<str>> = residencies
        .iter()
        .flat_map(|residency| {
            residency
                .prefixes
                .iter()
                .flat_map(|prefix| prefix.blocks.iter().map(|(hash, _)| hash.clone()))
        })
        .collect();
    let residency_owner_matches = residencies.iter().all(|residency| {
        residency.owner.target().instance_id.as_ref() == instance_id
            && residency.owner.target().dp_rank == dp_rank
    });
    if !residency_owner_matches {
        return Err(PortableSnapshotError::Invalid(
            "snapshot residency owner target mismatch",
        ));
    }
    let mut known = HashSet::<Arc<str>>::new();
    let mut pending: Vec<_> = residencies
        .iter()
        .flat_map(|residency| residency.prefixes.iter())
        .collect();
    while !pending.is_empty() {
        let before = pending.len();
        let mut next = Vec::new();
        for prefix in pending {
            let parent_known = prefix
                .parent_block_hash
                .as_ref()
                .is_none_or(|parent| known.contains(parent));
            if parent_known {
                known.extend(prefix.blocks.iter().map(|(hash, _)| hash.clone()));
            } else {
                next.push(prefix);
            }
        }
        if next.len() == before {
            return if next.iter().all(|prefix| {
                prefix
                    .parent_block_hash
                    .as_ref()
                    .is_some_and(|parent| all_hashes.contains(parent))
            }) {
                Err(PortableSnapshotError::Invalid(
                    "snapshot prefix parent cycle",
                ))
            } else {
                Err(PortableSnapshotError::Invalid(
                    "snapshot prefix has a dangling parent",
                ))
            };
        }
        pending = next;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> PortableCacheIdentity {
        PortableCacheIdentity::new("model", "tok", PortableHashMode::Sha256, 2)
    }

    fn fence() -> PortableSnapshotFence {
        PortableSnapshotFence::new("router-a", 1)
    }

    fn snapshot() -> PortableKvSnapshot {
        let mut snapshot = PortableKvSnapshot {
            format_version: PORTABLE_SNAPSHOT_FORMAT_VERSION,
            snapshot_id: "snapshot-1".into(),
            complete: true,
            cache_identity: identity(),
            created_at_unix_ms: 10_000,
            fence: fence(),
            checksum_sha256: String::new(),
            sources: vec![PortableSourceSnapshot {
                instance_id: "worker-0".into(),
                dp_rank: 0,
                incarnation: 1,
                sequence: 7,
                observed_at_unix_ms: 9_900,
                residencies: vec![PortableResidency {
                    owner: PortableResidencyOwner::Worker {
                        instance_id: "worker-0".into(),
                        dp_rank: 0,
                        incarnation: 1,
                    },
                    tier: PortableStorageTier::Device,
                    prefixes: vec![PortablePrefix {
                        parent_block_hash: None,
                        blocks: vec![
                            PortableBlock {
                                block_hash: "b:01".into(),
                                local_hash: "l:01".into(),
                            },
                            PortableBlock {
                                block_hash: "b:02".into(),
                                local_hash: "l:02".into(),
                            },
                        ],
                    }],
                }],
            }],
        };
        snapshot.seal().unwrap();
        snapshot
    }

    #[test]
    fn sealed_snapshot_validates_and_maps_residency() {
        let validated =
            validate_portable_snapshot(&snapshot(), &identity(), &fence(), 4, 16, 10_000).unwrap();
        assert_eq!(validated.sources.len(), 1);
        assert_eq!(validated.block_count, 2);
        assert_eq!(validated.sources[0].target.instance_id.as_ref(), "worker-0");
        assert_eq!(
            validated.sources[0].residencies[0].owner,
            ResidencyOwner::Worker {
                source: Arc::from("worker-0"),
                dp_rank: 0,
                incarnation: 1,
            }
        );
    }

    #[test]
    fn fence_is_sealed_and_must_match_the_trusted_expectation() {
        let mut tampered_fence = snapshot();
        tampered_fence.fence = PortableSnapshotFence::new("router-b", 1);
        assert_eq!(
            validate_portable_snapshot(
                &tampered_fence,
                &identity(),
                &PortableSnapshotFence::new("router-b", 1),
                4,
                16,
                10_000,
            )
            .unwrap_err(),
            PortableSnapshotError::ChecksumMismatch
        );

        assert_eq!(
            validate_portable_snapshot(
                &snapshot(),
                &identity(),
                &PortableSnapshotFence::new("router-b", 1),
                4,
                16,
                10_000,
            )
            .unwrap_err(),
            PortableSnapshotError::FenceMismatch
        );
        assert_eq!(
            validate_portable_snapshot(
                &snapshot(),
                &identity(),
                &PortableSnapshotFence::new("router-a", 2),
                4,
                16,
                10_000,
            )
            .unwrap_err(),
            PortableSnapshotError::FenceMismatch
        );
    }

    #[test]
    fn tamper_incomplete_identity_and_limits_are_rejected() {
        let mut tampered = snapshot();
        tampered.sources[0].sequence = 8;
        assert_eq!(
            validate_portable_snapshot(&tampered, &identity(), &fence(), 4, 16, 10_000)
                .unwrap_err(),
            PortableSnapshotError::ChecksumMismatch
        );

        let mut incomplete = snapshot();
        incomplete.complete = false;
        incomplete.seal().unwrap();
        assert_eq!(
            validate_portable_snapshot(&incomplete, &identity(), &fence(), 4, 16, 10_000)
                .unwrap_err(),
            PortableSnapshotError::Incomplete
        );

        let mut other_identity = identity();
        other_identity.tokenizer = "other-tokenizer".into();
        assert_eq!(
            validate_portable_snapshot(&snapshot(), &other_identity, &fence(), 4, 16, 10_000)
                .unwrap_err(),
            PortableSnapshotError::IdentityMismatch
        );
        assert_eq!(
            validate_portable_snapshot(&snapshot(), &identity(), &fence(), 4, 1, 10_000)
                .unwrap_err(),
            PortableSnapshotError::BlockBudgetExceeded
        );
    }

    #[test]
    fn dangling_and_cyclic_prefixes_are_rejected() {
        let mut dangling = snapshot();
        dangling.sources[0].residencies[0].prefixes[0].parent_block_hash = Some("b:missing".into());
        dangling.seal().unwrap();
        assert_eq!(
            validate_portable_snapshot(&dangling, &identity(), &fence(), 4, 16, 10_000)
                .unwrap_err(),
            PortableSnapshotError::Invalid("snapshot prefix has a dangling parent")
        );

        let mut cyclic = snapshot();
        cyclic.sources[0].residencies[0].prefixes = vec![
            PortablePrefix {
                parent_block_hash: Some("b:02".into()),
                blocks: vec![PortableBlock {
                    block_hash: "b:01".into(),
                    local_hash: "l:01".into(),
                }],
            },
            PortablePrefix {
                parent_block_hash: Some("b:01".into()),
                blocks: vec![PortableBlock {
                    block_hash: "b:02".into(),
                    local_hash: "l:02".into(),
                }],
            },
        ];
        cyclic.seal().unwrap();
        assert_eq!(
            validate_portable_snapshot(&cyclic, &identity(), &fence(), 4, 16, 10_000).unwrap_err(),
            PortableSnapshotError::Invalid("snapshot prefix parent cycle")
        );
    }

    #[test]
    fn worker_residency_must_use_the_source_incarnation() {
        let mut wrong_incarnation = snapshot();
        wrong_incarnation.sources[0].residencies[0].owner = PortableResidencyOwner::Worker {
            instance_id: "worker-0".into(),
            dp_rank: 0,
            incarnation: 2,
        };
        wrong_incarnation.seal().unwrap();
        assert_eq!(
            validate_portable_snapshot(&wrong_incarnation, &identity(), &fence(), 4, 16, 10_000)
                .unwrap_err(),
            PortableSnapshotError::Invalid(
                "worker residency incarnation does not match its snapshot source"
            )
        );
    }

    #[test]
    fn shared_owner_is_accepted_only_for_its_publishing_target() {
        let mut shared = snapshot();
        shared.sources[0].residencies[0].owner = PortableResidencyOwner::CacheOwner {
            pool_id: "pool-a".into(),
            source_instance_id: "worker-0".into(),
            dp_rank: 0,
        };
        shared.sources[0].residencies[0].tier = PortableStorageTier::HostPinned;
        shared.seal().unwrap();
        assert!(validate_portable_snapshot(&shared, &identity(), &fence(), 4, 16, 10_000).is_ok());

        shared.sources[0].residencies[0].owner = PortableResidencyOwner::CacheOwner {
            pool_id: "pool-a".into(),
            source_instance_id: "worker-1".into(),
            dp_rank: 0,
        };
        shared.seal().unwrap();
        assert_eq!(
            validate_portable_snapshot(&shared, &identity(), &fence(), 4, 16, 10_000).unwrap_err(),
            PortableSnapshotError::Invalid(
                "shared residency is attributed outside its snapshot source"
            )
        );
    }

    #[test]
    fn source_content_fingerprint_ignores_metadata_and_prefix_order() {
        let mut source = snapshot().sources.remove(0);
        let original = source_content_sha256(&source).unwrap();

        source.observed_at_unix_ms += 100;
        assert_eq!(source_content_sha256(&source).unwrap(), original);

        source.residencies[0].prefixes.push(PortablePrefix {
            parent_block_hash: None,
            blocks: vec![PortableBlock {
                block_hash: "b:03".into(),
                local_hash: "l:03".into(),
            }],
        });
        let with_second_prefix = source_content_sha256(&source).unwrap();
        source.residencies[0].prefixes.reverse();
        assert_eq!(source_content_sha256(&source).unwrap(), with_second_prefix);

        source.residencies[0].prefixes[0].blocks[0].local_hash = "different".into();
        assert_ne!(source_content_sha256(&source).unwrap(), with_second_prefix);
    }
}
