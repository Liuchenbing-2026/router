use axum::body::Body;
use axum::extract::Request;
use axum::http::HeaderMap;

pub use crate::otel_http::TRACE_HEADER_NAMES;

/// Client-facing protocol headers for the Anthropic Messages API.
///
/// `anthropic-version` is mandatory for `/v1/messages` callers and backends
/// reject requests that omit it, so the transparent proxy must forward it
/// verbatim instead of only injecting the router's own credential.
pub const ANTHROPIC_PROTOCOL_HEADER_NAMES: &[&str] = &["anthropic-version", "anthropic-beta"];

/// Copy request headers to a Vec of name-value string pairs
/// Used for forwarding headers to backend workers
pub fn copy_request_headers(req: &Request<Body>) -> Vec<(String, String)> {
    req.headers()
        .iter()
        .filter_map(|(name, value)| {
            // Convert header value to string, skipping non-UTF8 headers
            value
                .to_str()
                .ok()
                .map(|v| (name.to_string(), v.to_string()))
        })
        .collect()
}

/// Convert headers from reqwest Response to axum HeaderMap
/// Filters out hop-by-hop headers that shouldn't be forwarded
pub fn preserve_response_headers(reqwest_headers: &HeaderMap) -> HeaderMap {
    let mut headers = HeaderMap::new();

    for (name, value) in reqwest_headers.iter() {
        // Skip hop-by-hop headers that shouldn't be forwarded
        let name_str = name.as_str().to_lowercase();
        if should_forward_header(&name_str) {
            // The original name and value are already valid, so we can just clone them
            headers.append(name.clone(), value.clone());
        }
    }

    headers
}

/// Determine if a header should be forwarded from backend to client
fn should_forward_header(name: &str) -> bool {
    // List of headers that should NOT be forwarded (hop-by-hop headers)
    !matches!(
        name,
        "connection" |
        "keep-alive" |
        "proxy-authenticate" |
        "proxy-authorization" |
        "te" |
        "trailers" |
        "transfer-encoding" |
        "upgrade" |
        "content-encoding" | // Let axum/hyper handle encoding
        "host" // Should not forward the backend's host header
    )
}

/// Propagate OpenTelemetry trace headers to a reqwest RequestBuilder
///
/// When OTel is enabled: actively injects the current span's trace context,
/// making the router's span the parent of the backend request's span.
/// When OTel is disabled: passively forwards existing trace headers from
/// the incoming request.
pub fn propagate_trace_headers(
    request: reqwest::RequestBuilder,
    headers: Option<&HeaderMap>,
) -> reqwest::RequestBuilder {
    crate::otel_http::propagate_trace_headers(request, headers)
}

/// Propagate specific headers from incoming request to outgoing reqwest RequestBuilder
///
/// This is a general-purpose helper for selectively forwarding headers by name.
/// Only headers whose names match the provided list (case-insensitive) are propagated.
///
/// # Arguments
/// * `request` - The reqwest RequestBuilder to add headers to
/// * `headers` - Optional incoming headers to check
/// * `header_names` - List of header names to propagate (matched case-insensitively)
///
/// # Returns
/// The RequestBuilder with matching headers added
pub fn propagate_headers(
    mut request: reqwest::RequestBuilder,
    headers: Option<&HeaderMap>,
    header_names: &[&str],
) -> reqwest::RequestBuilder {
    if let Some(h) = headers {
        for &name in header_names {
            if let Some(value) = h.get(name) {
                request = request.header(name, value);
            }
        }
    }
    request
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preserve_repeated_set_cookie_headers() {
        let mut headers = HeaderMap::new();
        headers.append("set-cookie", "a=1; Path=/".parse().unwrap());
        headers.append("set-cookie", "b=2; Path=/".parse().unwrap());

        let result = preserve_response_headers(&headers);
        let cookies: Vec<_> = result
            .get_all("set-cookie")
            .iter()
            .map(|value| value.to_str().unwrap())
            .collect();

        assert_eq!(cookies, ["a=1; Path=/", "b=2; Path=/"]);
    }

    #[test]
    fn test_preserve_response_headers_filters_hop_by_hop_headers() {
        let mut headers = HeaderMap::new();
        headers.insert("content-type", "application/json".parse().unwrap());
        headers.append("connection", "keep-alive".parse().unwrap());
        headers.append("connection", "close".parse().unwrap());
        headers.insert("transfer-encoding", "chunked".parse().unwrap());

        let result = preserve_response_headers(&headers);

        assert_eq!(result.len(), 1);
        assert_eq!(result["content-type"], "application/json");
    }

    #[test]
    fn test_propagate_headers_forwards_anthropic_protocol_headers() {
        let mut headers = HeaderMap::new();
        headers.insert("anthropic-version", "2023-06-01".parse().unwrap());
        headers.insert(
            "anthropic-beta",
            "prompt-caching-2024-07-31".parse().unwrap(),
        );
        headers.insert("authorization", "Bearer secret".parse().unwrap());

        let request = propagate_headers(
            reqwest::Client::new().post("http://127.0.0.1:1/v1/messages"),
            Some(&headers),
            ANTHROPIC_PROTOCOL_HEADER_NAMES,
        )
        .build()
        .unwrap();

        assert_eq!(request.headers()["anthropic-version"], "2023-06-01");
        assert_eq!(
            request.headers()["anthropic-beta"],
            "prompt-caching-2024-07-31"
        );
        // Only protocol headers travel downstream; credentials stay router-side.
        assert!(request.headers().get("authorization").is_none());
    }
}
