from __future__ import annotations

from adapter.protocol import EnvAction
from agent.tools.sim_mcp import SimulatorMcpEpisodeConfig, SimulatorMcpEpisodeEnvironment
from extensions.gazebo import GazeboOracleMcpTransport


def test_openeta_mcp_episode_oracle_lifecycle_and_cleanup(tmp_path) -> None:
    transport = GazeboOracleMcpTransport()
    episode = SimulatorMcpEpisodeEnvironment(
        transport=transport,
        config=SimulatorMcpEpisodeConfig(
            env_id="openeta/gazebo_oracle-v0",
            seed=23,
            image_output_root=tmp_path,
        ),
    )
    try:
        initial = episode.reset(task="observe the industrial scene")
        assert initial.metadata["observation_provenance"] == "gazebo_oracle"
        assert initial.task == "observe the industrial scene"
        refreshed = episode.step(EnvAction(action_type="observe"))
        assert refreshed.observation.metadata["source"] == "SimulatorMcpEpisodeEnvironment"
        assert refreshed.info["environment_receipt_trusted"] is True
    finally:
        result = episode.close()
    assert result["ok"] is True
    assert transport.active_handles == ()
    assert transport.close_calls == 1
