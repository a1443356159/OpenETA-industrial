"""OpenETA local adapter facade for :class:`GazeboEnvironment`."""

from __future__ import annotations

from adapter.protocol import EnvAction, EnvObservation, StepResult
from adapter.sim import SimulatorAdapter

from .lifecycle import GazeboEnvironment, GazeboLifecycleError


class GazeboSimulatorAdapter(SimulatorAdapter):
    """Expose the M1 lifecycle through the standard simulator adapter API."""

    def __init__(self, environment: GazeboEnvironment | None = None) -> None:
        self.environment = environment or GazeboEnvironment()

    def reset(self, *, task: str | None = None, seed: int | None = None) -> EnvObservation:
        return self.environment.reset(task=task, seed=seed)

    def observe(self) -> EnvObservation:
        return self.environment.observe()

    def step(self, action: EnvAction) -> StepResult:
        raise GazeboLifecycleError(
            "Gazebo M1 adapter is read-only; manipulation/control is deferred to M2"
        )

    def close(self) -> None:
        self.environment.close()

