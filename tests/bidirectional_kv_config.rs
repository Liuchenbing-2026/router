use vllm_router_rs::config::{
    BidirectionalKvConfig, ConfigValidator, KvConnector, RouterConfig, RoutingMode,
};

fn config() -> RouterConfig {
    RouterConfig {
        bidirectional_kv: Some(BidirectionalKvConfig::default()),
        mode: RoutingMode::VllmPrefillDecode {
            prefill_urls: vec![("http://127.0.0.1:8100".into(), None)],
            decode_urls: vec!["http://127.0.0.1:8200".into()],
            prefill_policy: None,
            decode_policy: None,
            discovery_address: None,
        },
        ..RouterConfig::default()
    }
}

#[test]
fn disabled_by_default_and_static_pull_is_valid() {
    assert!(RouterConfig::default().bidirectional_kv.is_none());
    ConfigValidator::validate(&config()).unwrap();
    let mut mooncake_pull = config();
    mooncake_pull.kv_connector = KvConnector::MooncakePull;
    ConfigValidator::validate(&mooncake_pull).unwrap();
    assert!(serde_json::from_str::<BidirectionalKvConfig>(r#"{"misspelled_limit":8}"#).is_err());
}

#[test]
fn unsupported_topologies_and_zero_bounds_are_rejected() {
    let mut regular = config();
    regular.mode = RoutingMode::Regular {
        worker_urls: vec!["http://127.0.0.1:8000".into()],
    };
    let mut mooncake = config();
    mooncake.kv_connector = KvConnector::Mooncake;
    let mut dp = config();
    dp.intra_node_data_parallel_size = 2;
    let mut unbounded = config();
    unbounded.bidirectional_kv.as_mut().unwrap().max_entries = 0;
    let mut bytes = config();
    bytes.bidirectional_kv.as_mut().unwrap().max_bytes = 1;
    let mut ttl = config();
    ttl.bidirectional_kv.as_mut().unwrap().ttl_secs = 0;
    for case in [regular, mooncake, dp, unbounded, bytes, ttl] {
        assert!(ConfigValidator::validate(&case).is_err());
    }
}
