from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from agent.runtime.urdf_jacobian import UrdfSerialChain, capability_map_plugin


SIMPLE_URDF = """
<robot name="one_link">
  <link name="base"/>
  <link name="arm"/>
  <link name="tip"/>
  <joint name="joint_1" type="revolute">
    <parent link="base"/>
    <child link="arm"/>
    <axis xyz="0 0 1"/>
  </joint>
  <joint name="tool_offset" type="fixed">
    <parent link="arm"/>
    <child link="tip"/>
    <origin xyz="1 0 0"/>
  </joint>
</robot>
"""


def test_urdf_chain_computes_concrete_branch_jacobian_quality():
    chain = UrdfSerialChain.from_urdf(
        SIMPLE_URDF,
        base_link="base",
        tip_link="tip",
    )

    jacobian = chain.jacobian(["joint_1"], [0.0])

    assert chain.movable_joint_names == ("joint_1",)
    assert chain.translation_upper_bound_m == pytest.approx(1.0)
    assert chain.translation_lower_bound_m == pytest.approx(1.0)
    assert jacobian.shape == (6, 1)
    assert np.allclose(jacobian[:, 0], [0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    xyz, quaternion = chain.forward_kinematics(["joint_1"], [math.pi / 2.0])
    assert xyz == pytest.approx([0.0, 1.0, 0.0])
    assert quaternion == pytest.approx(
        [0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]
    )
    assert chain.minimum_singular_value(["joint_1"], [0.0]) == pytest.approx(
        math.sqrt(2.0)
    )


def test_urdf_chain_rejects_missing_or_nonfinite_joint_states():
    chain = UrdfSerialChain.from_urdf(
        SIMPLE_URDF,
        base_link="base",
        tip_link="tip",
    )

    with pytest.raises(ValueError, match="omitted"):
        chain.jacobian(["other"], [0.0])
    with pytest.raises(ValueError, match="finite"):
        chain.jacobian(["joint_1"], [float("nan")])


def test_builtin_capability_map_plugin_uses_expanded_urdf(tmp_path):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(SIMPLE_URDF, encoding="utf-8")
    plugin = capability_map_plugin(
        SimpleNamespace(
            urdf=urdf,
            base_link="base",
            tcp="tip",
            joint_lower=[-math.pi],
        )
    )

    xyz, quaternion = plugin.forward_kinematics(np.asarray([0.0]))

    assert xyz == pytest.approx([1.0, 0.0, 0.0])
    assert quaternion == pytest.approx([0.0, 0.0, 0.0, 1.0])
    assert plugin.jacobian(np.asarray([0.0])).shape == (6, 1)
