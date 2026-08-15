"""Evidence-only isolation probes shared by live Gazebo diagnostics.

This module deliberately avoids ros2cli graph listing.  ros2cli may start a
daemon and consequently changes the thing an acceptance run is trying to
observe.  The probe is run in a short lived child process with its own rclpy
context instead.
"""
from __future__ import annotations

from collections import Counter
from contextlib import suppress
import multiprocessing
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

PROBE_VERSION = "openeta.acceptance_isolation.v2"
PASSED, FAILED, INCONCLUSIVE = "PASSED", "FAILED", "INCONCLUSIVE"


def _normalise_node_names(raw: Any) -> list[dict[str, str]]:
    """Accept both the current rclpy row API and older two-column shapes."""
    if (
        isinstance(raw, tuple)
        and len(raw) == 2
        and all(isinstance(column, list) for column in raw)
        and all(isinstance(value, str) for column in raw for value in column)
    ):
        rows = zip(raw[0], raw[1])
    else:
        rows = raw
    return sorted(
        ({"name": str(name), "namespace": str(namespace)} for name, namespace in rows),
        key=lambda item: (item["namespace"], item["name"]),
    )


def _remove_own_node(
    nodes: list[dict[str, str]], *, name: str, namespace: str
) -> list[dict[str, str]]:
    """Remove exactly the observer instance, preserving same-name evidence."""
    result = list(nodes)
    with suppress(ValueError):
        result.remove({"name": name, "namespace": namespace})
    return result


def _normalise_graph_rows(raw: Any) -> list[list[Any]]:
    """Return ROS graph rows in a deterministic JSON-native representation.

    rclpy exposes topic/service/action rows as tuples.  The protected-domain
    baseline is persisted in JSON, where those same tuples are read back as
    lists.  Without normalisation, an unchanged graph compares unequal solely
    because one snapshot has ``tuple`` rows and the other has ``list`` rows.
    Preserve every endpoint and type while making both observations represent
    the exact same JSON topology.
    """

    rows: list[list[Any]] = []
    for row in raw:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ValueError("ROS graph row must be a two-column sequence")
        name, types = row
        if not isinstance(types, (list, tuple)):
            raise ValueError("ROS graph row types must be a sequence")
        rows.append([str(name), sorted(str(item) for item in types)])
    return sorted(rows, key=lambda item: (str(item[0]), item[1]))


def _graph_child(domain: int, connection: Any) -> None:
    started = time.monotonic()
    context = node = executor = None
    try:
        import rclpy
        from rclpy.action.graph import get_action_names_and_types
        from rclpy.executors import SingleThreadedExecutor

        context = rclpy.context.Context()
        context.init(domain_id=domain)
        node = rclpy.create_node(
            "openeta_acceptance_probe", context=context,
            enable_rosout=False, start_parameter_services=False,
        )
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)
        # Discovery is asynchronous; spin rather than merely sleeping.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
        # Keep duplicates: graph ownership is not inferred, but duplicate names
        # are evidence and must not disappear through a set conversion.
        nodes = _remove_own_node(
            _normalise_node_names(node.get_node_names_and_namespaces()),
            name=node.get_name(),
            namespace=node.get_namespace(),
        )
        connection.send({
            "availability": "AVAILABLE", "duration_ms": round((time.monotonic() - started) * 1000, 1),
            "nodes": nodes,
            "topics": _normalise_graph_rows(node.get_topic_names_and_types()),
            "services": _normalise_graph_rows(node.get_service_names_and_types()),
            "actions": _normalise_graph_rows(get_action_names_and_types(node=node)),
        })
    except BaseException as exc:  # child must turn all failures into evidence
        connection.send({
            "availability": "UNAVAILABLE", "duration_ms": round((time.monotonic() - started) * 1000, 1),
            "error_type": type(exc).__name__, "error": str(exc)[:500],
        })
    finally:
        try:
            if executor is not None and node is not None:
                executor.remove_node(node)
            if node is not None:
                node.destroy_node()
            if context is not None:
                context.shutdown()
        except BaseException:
            pass
        connection.close()


def probe_ros_graph(domain: int, timeout_s: float = 6.0) -> dict[str, Any]:
    """Return a structured snapshot; unavailability is never interpreted as empty."""
    started = time.monotonic()
    parent, child = multiprocessing.get_context("fork").Pipe(duplex=False)
    process = multiprocessing.get_context("fork").Process(target=_graph_child, args=(domain, child))
    process.start()
    child.close()
    payload: dict[str, Any]
    if parent.poll(timeout_s):
        payload = parent.recv()
        process.join(1.0)
    else:
        process.terminate()
        process.join(1.0)
        payload = {"availability": "UNAVAILABLE", "error_type": "Timeout", "error": "probe child exceeded hard timeout"}
    parent.close()
    payload["probe_version"] = PROBE_VERSION
    payload["domain"] = domain
    payload["parent_duration_ms"] = round((time.monotonic() - started) * 1000, 1)
    return payload


def node_multiset(snapshot: dict[str, Any]) -> Counter[tuple[str, str]]:
    return Counter((str(row.get("namespace", "")), str(row.get("name", ""))) for row in snapshot.get("nodes", []))


def empty_domain_evidence(domain: int, samples: int = 2) -> dict[str, Any]:
    observations = []
    for index in range(samples):
        observation = probe_ros_graph(domain)
        observations.append(observation)
        if observation.get("availability") != "AVAILABLE":
            return {"state": INCONCLUSIVE, "ok": None, "reason_code": "ROS_GRAPH_UNAVAILABLE", "observations": observations}
        if observation.get("nodes"):
            return {"state": FAILED, "ok": False, "reason_code": "ROS_DOMAIN_NOT_EMPTY", "observations": observations}
        if index + 1 < samples:
            time.sleep(0.5)
    return {"state": PASSED, "ok": True, "reason_code": "ROS_DOMAIN_EMPTY", "observations": observations}


def ros2cli_daemons(domain: int) -> list[dict[str, Any]]:
    """Find, but never terminate, daemons already tied to a candidate domain."""
    found = []
    expected = f"ROS_DOMAIN_ID={domain}".encode()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            environ = (proc / "environ").read_bytes().split(b"\0")
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if expected in environ and "ros2" in command and "daemon" in command:
            found.append({"pid": int(proc.name), "command": command[:500]})
    return sorted(found, key=lambda item: item["pid"])


def candidate_domain_evidence(domain: int) -> dict[str, Any]:
    daemons = ros2cli_daemons(domain)
    if daemons:
        return {"state": FAILED, "ok": False, "reason_code": "ROS2CLI_DAEMON_PRESENT", "daemons": daemons}
    return empty_domain_evidence(domain)


def gz_topics(*, environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    started = time.monotonic()
    try:
        result = subprocess.run(
            ["gz", "topic", "-l"], capture_output=True, text=True, timeout=8.0,
            check=False, env=dict(environment) if environment is not None else None,
        )
    except FileNotFoundError as exc:
        return {"availability": "UNAVAILABLE", "error_type": type(exc).__name__, "error": str(exc), "duration_ms": round((time.monotonic()-started)*1000, 1)}
    except subprocess.TimeoutExpired:
        return {"availability": "UNAVAILABLE", "error_type": "Timeout", "error": "gz topic -l timed out", "duration_ms": round((time.monotonic()-started)*1000, 1)}
    if result.returncode:
        return {"availability": "UNAVAILABLE", "error_type": "CommandFailed", "error": result.stderr.strip()[:500], "returncode": result.returncode, "duration_ms": round((time.monotonic()-started)*1000, 1)}
    return {"availability": "AVAILABLE", "topics": sorted(line.strip() for line in result.stdout.splitlines() if line.strip()), "duration_ms": round((time.monotonic()-started)*1000, 1)}


def world_partition_evidence(
    world_name: str, *, environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    result = gz_topics(environment=environment)
    if result["availability"] != "AVAILABLE":
        return {"state": INCONCLUSIVE, "ok": None, "reason_code": "GZ_GRAPH_UNAVAILABLE", "probe": result}
    topics = [item for item in result["topics"] if f"/world/{world_name}" in item]
    return {"state": PASSED if not topics else FAILED, "ok": not bool(topics), "reason_code": "GZ_PARTITION_EMPTY" if not topics else "GZ_WORLD_REMAINS", "world_topics": topics, "probe": result}


def aggregate_cleanup(checks: dict[str, dict[str, Any]]) -> str:
    states = {item.get("state") for item in checks.values()}
    if FAILED in states:
        return "failed"
    if INCONCLUSIVE in states:
        return "inconclusive"
    return "passed"
