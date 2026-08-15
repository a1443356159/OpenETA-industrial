from __future__ import annotations

import pytest

from extensions.gazebo.m2 import M2_ENV_ID, MODEL_ID, M2Config
from extensions.gazebo.m3 import (
    M3_DISPLAY_NAME,
    M3_ENV_ID,
    M3_MODEL_ID,
    M3_UNAVAILABLE_REASON,
    M3Config,
)
from extensions.gazebo.profiles import CONTROL, PHYSICS, STRUCTURED_RECEIPT, gazebo_profile
from sim.env_registry import get_env_spec


def test_m3_registration_is_explicitly_disabled_pending_detachable_joint() -> None:
    m3, m2 = get_env_spec(M3_ENV_ID), get_env_spec(M2_ENV_ID)
    assert m3 is not None and m2 is not None and m3.display_name == M3_DISPLAY_NAME
    assert M3Config().model_id == M3_MODEL_ID
    assert M2Config().model_id == MODEL_ID
    profile = gazebo_profile("m3_pickplace")
    assert profile.unavailable_reason == M3_UNAVAILABLE_REASON
    assert not ({CONTROL, PHYSICS, STRUCTURED_RECEIPT} & profile.capabilities)


def test_m3_scene_metadata_cannot_validate_or_start_manipulation_assets() -> None:
    with pytest.raises(RuntimeError, match=M3_UNAVAILABLE_REASON):
        M3Config().validate_assets()
