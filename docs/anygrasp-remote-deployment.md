# AnyGrasp remote deployment

> **Status:** Optional development-backend runbook. Final `multi_normal` acceptance uses GraspGenX
> and does not start or fall back to AnyGrasp.

This runbook deploys the real licensed AnyGrasp SDK used by OpenETA's
backend-neutral `grasp_pose_estimate` tool. It does not change the public tool
surface, bypass the MoveIt qualification funnel, or provide an acceptance-only
candidate path.

## Pinned inputs

| Component | Revision / digest |
|---|---|
| AnyGrasp SDK | `b8eaafc9eca7babd5208e7a5ade3c561060be4c5` |
| AnyGrasp modified MinkowskiEngine, `cuda-12-1` branch | `35862757e2586abc3b1528e012b6d80ce758a92c` |
| graspnetAPI | `eb57dd2092d8dbe05312a29c3d0c22f3226efbfc` |
| `checkpoint_detection.tar` | `a05c3690b95c8b65e78b1bb8a28f1d5ca96613391946e450afacae840bbcf7b2` |

The validated host uses Python 3.12.3, PyTorch `2.6.0+cu124`, the CUDA 12.6
toolkit, GCC 13.3, and an RTX 4090. Keep AnyGrasp and AnyPlace in separate
Python environments.

The service layout is:

```text
/root/autodl-tmp/openeta-services/anygrasp/
├── source/
│   └── grasp_detection/
│       ├── gsnet.so
│       └── license/
├── checkpoints/checkpoint_detection.tar
├── dependencies/
│   ├── MinkowskiEngine/
│   └── graspnetAPI/
└── venv/
```

## SDK license gate

The official SDK is machine licensed. For the current host, generate and
verify the feature id from `source/grasp_detection`:

```bash
cp gsnet_versions/gsnet.cpython-312-x86_64-linux-gnu.so gsnet.so
/root/autodl-tmp/openeta-services/anygrasp/venv/bin/python \
  -c 'from gsnet import get_feature_id; print(get_feature_id())'
```

The current machine reports `N66466733051868466746`. Place the issued files at
`source/grasp_detection/license/` and validate them before attempting model
inference:

```bash
/root/autodl-tmp/openeta-services/anygrasp/venv/bin/python \
  -c "from gsnet import check_license; check_license('license')"
```

Do not use the repository's `sample_license`: it is an example layout, not a
valid substitute. A missing or rejected license is an infrastructure blocker,
not permission to fall back while claiming an AnyGrasp test.

## CUDA build compatibility

The SDK's modified MinkowskiEngine needs two compatibility adjustments in this
environment:

1. PyTorch 2.6 already enables Ninja by default, but its
   `BuildExtension.with_options(use_ninja=True)` path forwards `use_ninja` into
   the Python 3.12/setuptools command constructor. In the deployed dependency's
   `setup.py`, use `cmdclass={"build_ext": BuildExtension}`. This retains Ninja
   while avoiding the invalid constructor argument.
2. CUDA 12.6 and GCC 13 produce the `std::__to_address` ambiguity documented by
   the AnyGrasp SDK for CUDA 12.8+ hosts. Back up
   `/usr/include/c++/13/bits/shared_ptr_base.h` as
   `shared_ptr_base.h.openeta-anygrasp.bak`, then qualify both occurrences of
   `__to_address(__r.get())` as `std::__to_address(__r.get())`. The validated
   backup SHA-256 is
   `e54d404d50842656ad03236a166f8bdf0fdd6badd17708e1a9b1911fce33612f`;
   the adjusted header SHA-256 is
   `a24f1f9b574798c80ada9bc7f3d39346d5c4b040ddcbe60f9046c4315f277206`.

Build for the 4090 only and use the system BLAS development package:

```bash
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=12
export PATH=/root/autodl-tmp/openeta-services/anygrasp/venv/bin:$PATH
cd /root/autodl-tmp/openeta-services/anygrasp/dependencies/MinkowskiEngine
python setup.py install \
  --blas=blas \
  --blas_include_dirs=/usr/include/x86_64-linux-gnu \
  --blas_library_dirs=/usr/lib/x86_64-linux-gnu
```

Install the SDK requirements, `pointnet2`, and graspnetAPI into the same
AnyGrasp environment. Keep NumPy at `1.26.4`, OpenCV at
`opencv-python==4.11.0.86`, scikit-image at `0.24.0`, and tifffile at
`2024.9.20`. The upstream graspnetAPI requirements are unbounded; resolving
them against the latest index otherwise upgrades NumPy across its major ABI
boundary. Reassert these pins after installing graspnetAPI and run `pip check`.
On hosts whose root pip cache is not writable, set `PIP_CACHE_DIR` below the
service directory. Verify both the PointNet2 and MinkowskiEngine CUDA paths
before launching the service.

## OpenETA service

`tools/anygrasp_mcp_server.py` resolves the SDK and checkpoint paths before
entering `source/grasp_detection`, because the official binary resolves
`license/licenseCfg.json` relative to its process working directory. This makes
stdio and service-manager launches independent of the caller's directory.

For the RM75/Robotiq 2F-85 profile, start the general service with the physical
opening bound and the unchanged 200-candidate reserve:

```bash
/root/autodl-tmp/openeta-services/anygrasp/venv/bin/python \
  tools/anygrasp_mcp_server.py \
  --transport sse \
  --host 127.0.0.1 \
  --port 8874 \
  --sdk-root /root/autodl-tmp/openeta-services/anygrasp/source \
  --checkpoint-path /root/autodl-tmp/openeta-services/anygrasp/checkpoints/checkpoint_detection.tar \
  --max-gripper-width 0.085 \
  --gripper-height 0.03 \
  --raw-pool-size 200
```

Set `OMP_NUM_THREADS=12` for this service so MinkowskiEngine does not select a
different implicit thread count when the deployment moves between hosts.

Health alone proves only that the MCP process is listening. A valid deployment
also runs the real sample inference test:

```bash
OPENETA_RUN_ANYGRASP_INTEGRATION=1 \
OPENETA_ANYGRASP_PYTHON=/root/autodl-tmp/openeta-services/anygrasp/venv/bin/python \
OPENETA_ANYGRASP_SDK_ROOT=/root/autodl-tmp/openeta-services/anygrasp/source \
OPENETA_ANYGRASP_SAMPLE_DIR=/root/autodl-tmp/openeta-services/anygrasp/source/grasp_detection/example_data \
OPENETA_ANYGRASP_CHECKPOINT_PATH=/root/autodl-tmp/openeta-services/anygrasp/checkpoints/checkpoint_detection.tar \
pytest -q tests/integration/test_anygrasp_mcp_server.py
```

Only after that smoke passes should normal acceptance run with `--anygrasp-url` and
`--anyplace-url`. The acceptance verifier still requires real model provenance,
full host qualification evidence, native contact/attach/detach, and stable
placement; changing the primary grasp backend does not relax those gates.
