"""Gazebo transport control plane for M3 bilateral-contact adhesion.

The plugin owns contact parsing and all capture decisions.  This adapter only
calls its documented Boolean/StringMsg services and normalizes their JSON
receipts; it never attempts a geometric or TF fallback.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from typing import Any, Mapping

from .m3 import AdhesionReceipt, AdhesionState, coerce_adhesion_receipt
from .process import GazeboProcessError


class GazeboM3AdhesionControl:
    """Drive the repository-owned ``M3AdhesionSystem`` service contract."""

    schema_version = "openeta.m3.adhesion.v1"
    arm_endpoint = "/m3/adhesion/arm_contact_window"
    capture_endpoint = "/m3/adhesion/capture"
    release_endpoint = "/m3/adhesion/release"
    state_endpoint = "/m3/adhesion/state"

    def __init__(
        self,
        *,
        gz_executable: str = "gz",
        timeout_ms: int = 3000,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.gz_executable = gz_executable
        self.timeout_ms = int(timeout_ms)
        self.environment = dict(environment) if environment is not None else None

    @staticmethod
    def _decode_reply(stdout: str) -> Mapping[str, Any]:
        """Parse ``gz.msgs.StringMsg`` CLI output without accepting text hints."""

        payload = stdout.strip()
        match = re.search(r'data:\s*"((?:\\.|[^"\\])*)"', payload, re.DOTALL)
        if match:
            try:
                payload = json.loads(f'"{match.group(1)}"')
            except json.JSONDecodeError as exc:
                raise GazeboProcessError("M3 adhesion returned an invalid StringMsg") from exc
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise GazeboProcessError("M3 adhesion returned non-JSON state") from exc
        if not isinstance(decoded, Mapping) or decoded.get("schema") != GazeboM3AdhesionControl.schema_version:
            raise GazeboProcessError("M3 adhesion returned an incompatible receipt schema")
        return decoded

    def _request(self, endpoint: str) -> AdhesionReceipt:
        executable = shutil.which(self.gz_executable) or self.gz_executable
        try:
            result = subprocess.run(
                [
                    executable,
                    "service",
                    "-s",
                    endpoint,
                    "--reqtype",
                    "gz.msgs.Boolean",
                    "--reptype",
                    "gz.msgs.StringMsg",
                    "--timeout",
                    str(self.timeout_ms),
                    "--req",
                    "data: true",
                ],
                capture_output=True,
                text=True,
                timeout=max(1.0, self.timeout_ms / 1000.0 + 2.0),
                env=self.environment,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GazeboProcessError(f"M3 adhesion request failed: {endpoint}") from exc
        if result.returncode != 0:
            raise GazeboProcessError(
                f"M3 adhesion request failed for {endpoint}: {result.stderr[-500:]}"
            )
        receipt = coerce_adhesion_receipt(self._decode_reply(result.stdout))
        if receipt is None:
            raise GazeboProcessError("M3 adhesion returned no receipt")
        return receipt

    def arm_contact_window(self) -> AdhesionReceipt:
        """Open the plugin's post-close bilateral-contact evidence window."""

        receipt = self._request(self.arm_endpoint)
        if receipt.state not in {AdhesionState.ARMED, AdhesionState.CAPTURED}:
            raise GazeboProcessError(f"M3 adhesion refused contact window: {receipt.state.value}")
        return receipt

    def state(self) -> AdhesionReceipt:
        return self._request(self.state_endpoint)

    def _await_terminal(self, initial: AdhesionReceipt) -> AdhesionReceipt:
        deadline = time.monotonic() + self.timeout_ms / 1000.0
        receipt = initial
        while receipt.state in {
            AdhesionState.ARMED,
            AdhesionState.RELEASE_PENDING,
            AdhesionState.UNKNOWN,
        }:
            if time.monotonic() >= deadline:
                return AdhesionReceipt(
                    state=AdhesionState.REJECTED,
                    reason="CAPTURE_STATUS_TIMEOUT",
                    window_id=receipt.window_id,
                )
            time.sleep(0.03)
            receipt = self.state()
        return receipt

    def capture(self) -> AdhesionReceipt:
        """Ask the plugin to revalidate and capture its one native candidate."""

        return self._await_terminal(self._request(self.capture_endpoint))

    def release(self) -> AdhesionReceipt:
        """Release a capture after the user-visible gripper open action."""

        receipt = self._request(self.release_endpoint)
        if receipt.state in {AdhesionState.ARMED, AdhesionState.RELEASE_PENDING}:
            return self._await_terminal(receipt)
        return receipt
