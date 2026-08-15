from __future__ import annotations

import pytest

from extensions.gazebo.m3 import M3_UNAVAILABLE_REASON, M3Config


def test_m3_scene_contract_is_fail_closed_without_a_verifier() -> None:
    with pytest.raises(RuntimeError, match=M3_UNAVAILABLE_REASON):
        M3Config().validate_assets()
