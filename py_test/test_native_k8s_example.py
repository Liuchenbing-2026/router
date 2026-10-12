"""Keep the native example's engine, worker identities and discovery consistent."""

import os
import subprocess
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def native_example():
    return Path(__file__).resolve().parents[1] / "scripts/k8s/llama3/vllm-router/native"


@pytest.fixture
def native_config(native_example):
    values = yaml.safe_load((native_example / "values.yaml").read_text())
    manifests = list(
        yaml.safe_load_all((native_example / "router-deployment.yaml").read_text())
    )
    deployment = next(item for item in manifests if item["kind"] == "Deployment")
    return values, manifests, deployment


def option(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


@pytest.mark.parametrize(
    "field,flag",
    [("tensor", "--tensor-parallel-size"), ("data", "--data-parallel-size")],
)
def test_engine_parallelism_matches_chart(native_config, field, flag):
    values, _, _ = native_config
    engine = values["decode"]["containers"][0]
    assert (
        int(option(engine["args"], flag, "1")) == values["decode"]["parallelism"][field]
    )


def test_engine_uses_engine_cli_and_fits_allocated_gpus(native_config):
    values, _, _ = native_config
    engine = values["decode"]["containers"][0]
    assert "--intra-node-data-parallel-size" not in engine["args"]
    parallelism = values["decode"]["parallelism"]
    assert parallelism["tensor"] * parallelism["data"] == int(
        engine["resources"]["limits"]["nvidia.com/gpu"]
    )


def test_router_creates_only_real_dp_ranks(native_config):
    values, _, deployment = native_config
    router = deployment["spec"]["template"]["spec"]["containers"][0]
    assert int(option(router["command"], "--intra-node-data-parallel-size", "1")) == (
        values["decode"]["parallelism"]["data"]
    )


def test_router_discovers_pod_identity_at_engine_port(native_config):
    values, _, deployment = native_config
    argv = deployment["spec"]["template"]["spec"]["containers"][0]["command"]
    assert "--service-discovery" in argv
    assert "--worker-urls" not in argv
    assert (
        option(argv, "--service-discovery-namespace")
        == deployment["metadata"]["namespace"]
    )
    engine = values["decode"]["containers"][0]
    assert option(argv, "--service-discovery-port") == option(engine["args"], "--port")
    start = argv.index("--selector") + 1
    selectors = []
    for value in argv[start:]:
        if value.startswith("--"):
            break
        selectors.append(value)
    assert set(selectors) == {"llm-d.ai/role=decode", "llm-d.ai/inferenceServing=true"}


def test_discovery_rbac_is_namespaced_and_read_only(native_config):
    _, manifests, deployment = native_config
    resources = {item["kind"]: item for item in manifests}
    namespace = deployment["metadata"]["namespace"]
    account = resources["ServiceAccount"]
    role = resources["Role"]
    binding = resources["RoleBinding"]
    assert all(item["metadata"]["namespace"] == namespace for item in manifests)
    assert (
        deployment["spec"]["template"]["spec"]["serviceAccountName"]
        == account["metadata"]["name"]
    )
    assert role["rules"] == [
        {"apiGroups": [""], "resources": ["pods"], "verbs": ["get", "list", "watch"]}
    ]
    assert binding["roleRef"] == {
        "apiGroup": "rbac.authorization.k8s.io",
        "kind": "Role",
        "name": role["metadata"]["name"],
    }
    assert binding["subjects"] == [
        {
            "kind": "ServiceAccount",
            "name": account["metadata"]["name"],
            "namespace": namespace,
        }
    ]


def test_deployment_stops_when_backend_readiness_fails(native_example, tmp_path):
    command_log = tmp_path / "commands.txt"
    kubectl = tmp_path / "kubectl"
    kubectl.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$NATIVE_TEST_COMMAND_LOG"\n'
        'if [ "$1" = wait ]; then exit 1; fi\nexit 0\n'
    )
    kubectl.chmod(0o755)
    helmfile = tmp_path / "helmfile"
    helmfile.write_text("#!/bin/sh\nexit 0\n")
    helmfile.chmod(0o755)
    env = dict(
        os.environ,
        PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
        NATIVE_TEST_COMMAND_LOG=str(command_log),
    )
    result = subprocess.run(
        ["bash", str(native_example / "deploy.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "router-deployment.yaml" not in command_log.read_text()
