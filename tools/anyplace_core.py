"""AnyPlace placement backend for the OpenETA AnyPlace MCP server."""

from __future__ import annotations

import base64
import contextlib
import io
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.candidate_config import DEFAULT_CANDIDATE_COUNT, candidate_count as validate_candidate_count


MODEL_NAME = "anyplace_multitask"
FRAME = "placement_camera"
CAMERA_FRAME = "opencv"
POSE_CONVENTION = "p_placed = R @ p_current + t"
DEFAULT_CANDIDATE_LIMIT = DEFAULT_CANDIDATE_COUNT
MODEL_SAMPLE_COUNT = 1024
DEFAULT_DEPTH_TRUNCATION = 1.0


class AnyPlaceInputError(Exception):
    """Input or normalized-output data cannot satisfy the AnyPlace contract."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class PointCloudLimits:
    """Point-count limits for one AnyPlace point cloud input."""

    min_points: int
    max_points: int


# ``N_crop`` inside the official AnyPlace policy is 1024.  It is the tensor
# sample size, not a requirement for 1024 distinct sensor pixels: a held
# object can legitimately occupy fewer pixels in a calibrated placement view.
# Keep a non-trivial measured-point floor, then deterministically pad only the
# model input to its fixed sample size below.
MIN_MEASURED_POINTCLOUD_POINTS = 128
OBJECT_POINTCLOUD_LIMITS = PointCloudLimits(
    min_points=MIN_MEASURED_POINTCLOUD_POINTS, max_points=200000
)
PLACEMENT_REGION_POINTCLOUD_LIMITS = PointCloudLimits(
    min_points=MIN_MEASURED_POINTCLOUD_POINTS, max_points=500000
)


class _NoOpVisualizer:
    """Meshcat-compatible sink for official AnyPlace helpers that always draw."""

    def __getitem__(self, _name: str) -> "_NoOpVisualizer":
        return self

    def set_object(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def set_transform(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class AnyPlaceBackend:
    """Lazy in-process AnyPlace backend.

    This wrapper intentionally does not invoke official CLI scripts. It loads
    the AnyPlace model inside the MCP server process and calls the official
    internal policy function directly.
    """

    def __init__(
        self,
        *,
        anyplace_root: str | Path,
        config_path: str | Path,
        seed: int = 0,
        depth_truncation: float = DEFAULT_DEPTH_TRUNCATION,
        candidate_count: int = DEFAULT_CANDIDATE_LIMIT,
    ) -> None:
        self.anyplace_root = Path(anyplace_root)
        self.config_path = Path(config_path)
        self.seed = seed
        self.depth_truncation = depth_truncation
        self.candidate_count = validate_candidate_count(candidate_count)
        self._loaded: dict[str, Any] | None = None
        self._prediction_count = 0

    def predict_placement(
        self,
        *,
        object_observation: dict[str, Any] | None,
        placement_observation: dict[str, Any] | None,
        object_camera_to_placement_camera: list[list[float]] | None,
        placement_camera_to_world: list[list[float]] | None,
    ) -> dict[str, Any]:
        """Predict object-goal transforms from independent RGB-D observations."""

        start = time.perf_counter()
        metadata = _metadata_base()
        inference_seed = self.seed + self._prediction_count
        self._prediction_count += 1
        metadata["inference_seed"] = inference_seed
        try:
            np, Image = _load_numeric_deps()
            object_packet = _observation_packet(object_observation, "object")
            placement_packet = _observation_packet(placement_observation, "placement")
            object_rgb = _decode_image_payload(
                object_packet["rgb"],
                Image=Image,
                np=np,
                convert="RGB",
                missing_reason="missing_object_rgb",
                decode_reason="object_rgb_decode_failed",
            )
            object_depth = _decode_image_payload(
                object_packet["depth"],
                Image=Image,
                np=np,
                convert=None,
                missing_reason="missing_object_depth",
                decode_reason="object_depth_decode_failed",
            )
            object_mask_array = _decode_image_payload(
                object_packet["mask"],
                Image=Image,
                np=np,
                convert="L",
                missing_reason="missing_object_mask",
                decode_reason="object_mask_decode_failed",
            )
            placement_rgb = _decode_image_payload(
                placement_packet["rgb"], Image=Image, np=np, convert="RGB",
                missing_reason="missing_placement_rgb", decode_reason="placement_rgb_decode_failed",
            )
            placement_depth = _decode_image_payload(
                placement_packet["depth"], Image=Image, np=np, convert=None,
                missing_reason="missing_placement_depth", decode_reason="placement_depth_decode_failed",
            )
            placement_mask_array = _decode_image_payload(
                placement_packet["mask"],
                Image=Image,
                np=np,
                convert="L",
                missing_reason="missing_placement_region_mask",
                decode_reason="placement_region_mask_decode_failed",
            )
            object_pcd = build_masked_pointcloud_from_rgbd(
                rgb=object_rgb, depth=object_depth, mask=object_mask_array,
                intrinsics=validate_intrinsics(object_packet["intrinsics"]),
                limits=OBJECT_POINTCLOUD_LIMITS,
                empty_reason="empty_object_pointcloud",
                too_small_reason="object_pointcloud_too_small",
                depth_truncation=self.depth_truncation,
            )
            placement_pcd = build_masked_pointcloud_from_rgbd(
                rgb=placement_rgb, depth=placement_depth, mask=placement_mask_array,
                intrinsics=validate_intrinsics(placement_packet["intrinsics"]),
                limits=PLACEMENT_REGION_POINTCLOUD_LIMITS,
                empty_reason="empty_placement_region_pointcloud",
                too_small_reason="placement_region_pointcloud_too_small",
                depth_truncation=self.depth_truncation,
            )
            transform = np.asarray(object_camera_to_placement_camera, dtype=np.float64)
            if not _is_rigid_transform(transform):
                raise AnyPlaceInputError("invalid_observation_transform")
            placement_to_world = np.asarray(
                placement_camera_to_world, dtype=np.float64
            )
            if not _is_rigid_transform(placement_to_world):
                raise AnyPlaceInputError("invalid_placement_camera_to_world")
            object_h = np.concatenate(
                [object_pcd.astype(np.float64), np.ones((object_pcd.shape[0], 1))], axis=1
            )
            object_pcd = (transform @ object_h.T).T[:, :3].astype(np.float32)
            # The official AnyPlace policy was trained on gravity-aligned world
            # point clouds (its documented output transforms child/B "in the
            # world frame").  OpenCV optical axes instead have +Z forward and
            # +Y down.  Giving raw camera clouds to the policy reverses its
            # notion of vertical and produces goals below a horizontal support.
            # Keep the public result in placement-camera coordinates by
            # conjugating the policy transforms back below.
            object_h = np.concatenate(
                [object_pcd.astype(np.float64), np.ones((object_pcd.shape[0], 1))], axis=1
            )
            placement_h = np.concatenate(
                [placement_pcd.astype(np.float64), np.ones((placement_pcd.shape[0], 1))], axis=1
            )
            object_pcd = (placement_to_world @ object_h.T).T[:, :3].astype(np.float32)
            placement_pcd = (placement_to_world @ placement_h.T).T[:, :3].astype(np.float32)
            measured_object_pcd = object_pcd.copy()
            measured_placement_pcd = placement_pcd.copy()
            metadata.update(
                {
                    "object_point_count": int(object_pcd.shape[0]),
                    "placement_region_point_count": int(placement_pcd.shape[0]),
                    "model_frame": "world_gravity_aligned",
                }
            )
            object_pcd = pad_pointcloud_for_model(
                object_pcd, target_count=MODEL_SAMPLE_COUNT
            )
            placement_pcd = pad_pointcloud_for_model(
                placement_pcd, target_count=MODEL_SAMPLE_COUNT
            )
            metadata.update(
                {
                    "object_model_point_count": int(object_pcd.shape[0]),
                    "placement_region_model_point_count": int(placement_pcd.shape[0]),
                }
            )
        except AnyPlaceInputError as exc:
            return _failure_result(
                reason=exc.reason,
                content=f"AnyPlace placement prediction failed: {exc.reason}.",
                metadata=_with_duration(metadata, start),
            )

        try:
            backend = self._get_loaded_backend()
        except AnyPlaceInputError as exc:
            return _failure_result(
                reason=exc.reason,
                content=f"AnyPlace placement prediction failed: {exc.reason}.",
                metadata=_with_duration(metadata, start),
            )
        except Exception as exc:  # noqa: BLE001
            return _failure_result(
                reason="model_load_failed",
                content=f"AnyPlace placement prediction failed: model load failed: {exc}",
                metadata=_with_duration(
                    {**metadata, "error_type": type(exc).__name__},
                    start,
                ),
            )

        try:
            raw_candidates = self._predict_with_loaded_backend(
                backend=backend,
                object_pcd=object_pcd,
                placement_region_pcd=placement_pcd,
                inference_seed=inference_seed,
            )
            raw_candidates = _project_object_bottoms_to_support(
                raw_candidates,
                object_points=measured_object_pcd,
                support_points=measured_placement_pcd,
                np=np,
            )
            raw_candidates = _model_world_to_placement_camera_transforms(
                raw_candidates, placement_camera_to_world=placement_to_world, np=np
            )
            candidates = normalise_placement_candidates(
                raw_candidates,
                expected_count=self.candidate_count,
            )
        except AnyPlaceInputError as exc:
            return _failure_result(
                reason=exc.reason,
                content=f"AnyPlace placement prediction failed: {exc.reason}.",
                metadata=_with_duration(metadata, start),
            )
        except Exception as exc:  # noqa: BLE001
            return _failure_result(
                reason="model_inference_failed",
                content=f"AnyPlace placement prediction failed: model inference failed: {exc}",
                metadata=_with_duration(
                    {**metadata, "error_type": type(exc).__name__},
                    start,
                ),
            )

        return {
            "success": True,
            "content": "AnyPlace placement prediction completed.",
            "details": {
                "tool": "anyplace",
                "backend": "anyplace_mcp",
                "model": MODEL_NAME,
                "frame": FRAME,
                "camera_frame": CAMERA_FRAME,
                "candidate_count": len(candidates),
                "placement_candidates": candidates,
                "metadata": _with_duration(
                    {**metadata, "configured_candidate_count": self.candidate_count},
                    start,
                ),
            },
        }

    def _get_loaded_backend(self) -> dict[str, Any]:
        if self._loaded is not None:
            return self._loaded
        if not self.anyplace_root.exists():
            raise RuntimeError(f"anyplace root does not exist: {self.anyplace_root}")
        if not self.config_path.exists():
            raise RuntimeError(f"config path does not exist: {self.config_path}")

        if str(self.anyplace_root) not in sys.path:
            sys.path.insert(0, str(self.anyplace_root))
        anyplace_source_dir = self.anyplace_root / "anyplace"
        os.environ["ANYPLACE_SOURCE_DIR"] = str(anyplace_source_dir)
        os.environ.setdefault("ANYPLACE_DATA_DIR", str(anyplace_source_dir))

        import torch

        if not torch.cuda.is_available():
            raise AnyPlaceInputError("device_unavailable")
        _configure_cuda_extension_environment(torch)

        import numpy as np
        from anyplace.model.transformer.policy import NSMTransformerImplicit
        from anyplace.model.transformer.policy import NSMTransformerSingleTransformationRegression
        from anyplace.utils import config_util, util
        from anyplace.utils.anyplace.multistep_pose_regression_anyplace import (
            policy_inference_methods_dict,
        )
        from anyplace.utils.mesh_util import three_util

        args = config_util.load_config(str(self.config_path), demo_train_eval="eval")
        args = config_util.recursive_attr_dict(args)
        args.experiment.eval.init_k_val = self.candidate_count
        ckpt_path = Path(args.experiment.eval.ckpt_path)
        if not ckpt_path.is_absolute():
            ckpt_path = self.config_path.parent / ckpt_path
        if not ckpt_path.exists():
            raise RuntimeError(f"checkpoint path does not exist: {ckpt_path}")

        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        with contextlib.redirect_stdout(sys.stderr):
            pose_refine_ckpt = torch.load(
                str(ckpt_path),
                map_location=torch.device("cpu"),
                weights_only=False,
            )
            config_util.update_recursive(
                args.model.refine_pose,
                config_util.recursive_attr_dict(pose_refine_ckpt["args"]["model"]["refine_pose"]),
            )
            pr_type = args.model.refine_pose.type
            pr_args = config_util.copy_attr_dict(args.model[pr_type])
            if args.model.refine_pose.get("model_kwargs") is not None:
                custom_pr_args = args.model.refine_pose.model_kwargs[pr_type]
                config_util.update_recursive(pr_args, custom_pr_args)

            if pr_type == "nsm_transformer":
                pr_model_cls = NSMTransformerSingleTransformationRegression
                pr_model = pr_model_cls(
                    mc_vis=_NoOpVisualizer(),
                    feat_dim=args.model.refine_pose.feat_dim,
                    **pr_args,
                ).cuda()
            elif pr_type == "nsm_implicit":
                pr_model_cls = NSMTransformerImplicit
                pr_model = pr_model_cls(
                    mc_vis=_NoOpVisualizer(),
                    feat_dim=args.model.refine_pose.feat_dim,
                    is_train=False,
                    **pr_args,
                ).cuda()
            else:
                raise RuntimeError(f"unsupported AnyPlace refine pose type: {pr_type}")
            pr_model.load_state_dict(pose_refine_ckpt["refine_pose_model_state_dict"])
            pr_model.eval()
            if hasattr(pr_model, "eval_sample"):
                pr_model.set_eval_sample(True)

        reso_grid = args.data.voxel_grid.reso_grid
        padding_grid = args.data.voxel_grid.padding
        raster_pts = three_util.get_raster_points(reso_grid, padding=padding_grid)
        raster_pts = raster_pts.reshape(reso_grid, reso_grid, reso_grid, 3)
        raster_pts = raster_pts.transpose(2, 1, 0, 3)
        raster_pts = raster_pts.reshape(-1, 3)
        rot_grid = util.generate_healpix_grid(size=args.data.rot_grid_samples)
        args.data.rot_grid_bins = rot_grid.shape[0]
        scene_extents = np.asarray(args.data.coarse_aff.scene_extents, dtype=np.float64)
        args.data.coarse_aff.scene_scale = 1.0 / float(np.max(scene_extents))

        infer_relation_policy = policy_inference_methods_dict[
            args.experiment.eval.inference_method
        ]

        self._loaded = {
            "args": args,
            "pr_model": pr_model,
            "raster_pts": raster_pts,
            "rot_grid": rot_grid,
            "infer_relation_policy": infer_relation_policy,
            "torch": torch,
        }
        return self._loaded
    def _predict_with_loaded_backend(
        self,
        *,
        backend: dict[str, Any],
        object_pcd: Any,
        placement_region_pcd: Any,
        inference_seed: int,
    ) -> Any:
        args = backend["args"]
        infer_relation_policy = backend["infer_relation_policy"]
        torch = backend["torch"]
        exp_args = args.experiment
        infer_kwargs: dict[str, Any] = {
            "gt_child_cent": None,
            "export_viz": False,
            "export_viz_dirname": None,
            "export_viz_relative_trans_guess": None,
            "compute_coverage_scores": False,
            "out_coverage_dirname1": None,
            "out_coverage_dirname2": None,
            "iteration": 0,
        }
        if getattr(exp_args.eval, "multi_aff_rot", False):
            infer_kwargs["multi_aff_rot"] = True

        multi_mesh_dict = {
            "parent_file": None,
            "parent_scale": None,
            "parent_pose": None,
            "child_file": None,
            "child_scale": None,
            "child_pose": None,
            "multi": True,
        }

        torch.manual_seed(inference_seed)
        random.seed(inference_seed)
        with torch.no_grad(), contextlib.redirect_stdout(sys.stderr):
            relative_trans_preds = infer_relation_policy(
                _NoOpVisualizer(),
                placement_region_pcd,
                object_pcd,
                None,
                backend["pr_model"],
                None,
                scene_mean=args.data.coarse_aff.scene_mean,
                scene_scale=args.data.coarse_aff.scene_scale,
                grid_pts=backend["raster_pts"],
                rot_grid=backend["rot_grid"],
                viz=False,
                n_iters=exp_args.eval.n_refine_iters,
                no_parent_crop=(not exp_args.parent_crop),
                return_top=(not exp_args.eval.return_rand),
                with_coll=exp_args.eval.with_coll,
                run_affordance=exp_args.eval.run_affordance,
                init_k_val=self.candidate_count,
                no_sc_score=exp_args.eval.no_success_classifier,
                init_parent_mean=exp_args.eval.init_parent_mean_pos,
                init_orig_ori=exp_args.eval.init_orig_ori,
                refine_anneal=exp_args.eval.refine_anneal,
                mesh_dict=multi_mesh_dict,
                add_per_iter_noise=exp_args.eval.add_per_iter_noise,
                per_iter_noise_kwargs=exp_args.eval.per_iter_noise_kwargs,
                variable_size_crop=exp_args.eval.variable_size_crop,
                timestep_emb_decay_factor=exp_args.eval.timestep_emb_decay_factor,
                remove_redundant_pose=exp_args.eval.remove_redundant_pose,
                **infer_kwargs,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        return relative_trans_preds


def _configure_cuda_extension_environment(torch: Any) -> None:
    """Make AnyPlace's legacy JIT extensions reproducible in an isolated env.

    MCP stdio launches preserve only a conservative environment subset, so a
    Conda-local CUDA compiler cannot rely on caller-provided ``CC``/``CXX``.
    Prefer tools shipped beside the backend interpreter when they exist. CUDA
    11.7 also cannot emit native Ada (sm_89) code; PTX for sm_86 is the newest
    forward-compatible target supported by that compiler.
    """

    # Keep the interpreter's *invocation* directory.  A virtualenv's
    # ``python`` is normally a symlink to the system interpreter; resolving
    # it here would incorrectly turn ``<venv>/bin/python`` into
    # ``/usr/bin/python`` and hide sibling tools such as ``ninja``.
    prefix = Path(sys.executable).parent.parent
    bin_dir = prefix / "bin"
    nvcc = bin_dir / "nvcc"
    cc = bin_dir / "x86_64-conda-linux-gnu-gcc"
    cxx = bin_dir / "x86_64-conda-linux-gnu-g++"
    ninja = bin_dir / "ninja"
    if nvcc.is_file():
        os.environ.setdefault("CUDA_HOME", str(prefix))
    if cc.is_file():
        os.environ.setdefault("CC", str(cc))
    if cxx.is_file():
        os.environ.setdefault("CXX", str(cxx))
    if ninja.is_file():
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        if str(bin_dir) not in path_entries:
            os.environ["PATH"] = os.pathsep.join((str(bin_dir), *path_entries))

    cuda_version = str(getattr(torch.version, "cuda", "") or "")
    if cuda_version.startswith("11.7"):
        capability = tuple(int(item) for item in torch.cuda.get_device_capability())
        if capability >= (8, 9):
            os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6+PTX")


def validate_intrinsics(intrinsics: dict[str, Any] | None) -> dict[str, float]:
    if not isinstance(intrinsics, dict):
        raise AnyPlaceInputError("missing_intrinsics")
    parsed: dict[str, float] = {}
    for key in ("fx", "fy", "cx", "cy", "scale"):
        if key not in intrinsics:
            raise AnyPlaceInputError("invalid_intrinsics")
        try:
            value = float(intrinsics[key])
        except (TypeError, ValueError) as exc:
            raise AnyPlaceInputError("invalid_intrinsics") from exc
        if key in {"fx", "fy", "scale"} and value <= 0:
            raise AnyPlaceInputError("invalid_intrinsics")
        parsed[key] = value
    return parsed


def build_masked_pointcloud_from_rgbd(
    *,
    rgb: Any,
    depth: Any,
    mask: Any,
    intrinsics: dict[str, float],
    limits: PointCloudLimits,
    empty_reason: str,
    too_small_reason: str,
    depth_truncation: float,
) -> Any:
    """Project one aligned RGB-D mask into a camera-frame point cloud."""

    np, _Image = _load_numeric_deps()
    rgb_array = np.asarray(rgb)
    depth_array = np.asarray(depth)
    mask_2d = np.asarray(mask) > 0

    if rgb_array.ndim != 3 or rgb_array.shape[2] < 3:
        raise AnyPlaceInputError("image_shape_mismatch")
    if depth_array.ndim != 2:
        raise AnyPlaceInputError("unsupported_depth_format")
    if rgb_array.shape[:2] != depth_array.shape:
        raise AnyPlaceInputError("image_shape_mismatch")
    if mask_2d.shape != depth_array.shape:
        raise AnyPlaceInputError("image_shape_mismatch")
    if not mask_2d.any():
        raise AnyPlaceInputError(empty_reason.replace("pointcloud", "mask"))

    height, width = depth_array.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    points_z = depth_array.astype(np.float32) / float(intrinsics["scale"])
    points_x = (u - intrinsics["cx"]) / intrinsics["fx"] * points_z
    points_y = (v - intrinsics["cy"]) / intrinsics["fy"] * points_z
    points = np.stack([points_x, points_y, points_z], axis=-1)
    valid_depth = (points_z > 0) & (points_z < depth_truncation)

    return validate_pointcloud_array(
        points[valid_depth & mask_2d].astype(np.float32), limits=limits,
        empty_reason=empty_reason, too_small_reason=too_small_reason,
    )


def _observation_packet(value: Any, kind: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AnyPlaceInputError(f"missing_{kind}_observation")
    mask_key = "object_mask" if kind == "object" else "placement_region_mask"
    required = {"rgb", "depth", mask_key, "intrinsics"}
    if not required <= value.keys():
        raise AnyPlaceInputError(f"invalid_{kind}_observation")
    return {**value, "mask": value[mask_key]}


def validate_pointcloud_array(
    array: Any,
    *,
    limits: PointCloudLimits,
    empty_reason: str = "empty_pointcloud",
    too_small_reason: str = "pointcloud_too_small",
) -> Any:
    np, _Image = _load_numeric_deps()
    if not isinstance(array, np.ndarray):
        raise AnyPlaceInputError("invalid_pointcloud_shape")
    if array.ndim != 2 or array.shape[1] != 3:
        raise AnyPlaceInputError("invalid_pointcloud_shape")
    point_count = int(array.shape[0])
    if point_count == 0:
        raise AnyPlaceInputError(empty_reason)
    if point_count < limits.min_points:
        raise AnyPlaceInputError(too_small_reason)
    if point_count > limits.max_points:
        raise AnyPlaceInputError("request_too_large")
    if array.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise AnyPlaceInputError("invalid_pointcloud_shape")
    if not np.isfinite(array).all():
        raise AnyPlaceInputError("invalid_pointcloud_shape")
    return array.astype(np.float32, copy=False)


def pad_pointcloud_for_model(array: Any, *, target_count: int = MODEL_SAMPLE_COUNT) -> Any:
    """Pad a validated cloud to AnyPlace's fixed token count deterministically.

    This never manufactures geometry: it only repeats existing, measured
    points when the visible object/region has fewer samples than the official
    policy's fixed ``N_crop``.  Callers retain the measured count in metadata.
    """
    np, _Image = _load_numeric_deps()
    if not isinstance(array, np.ndarray) or array.ndim != 2 or array.shape[1] != 3:
        raise AnyPlaceInputError("invalid_pointcloud_shape")
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    count = int(array.shape[0])
    if count == 0:
        raise AnyPlaceInputError("empty_pointcloud")
    if count >= target_count:
        return array.astype(np.float32, copy=False)
    indices = np.linspace(0, count - 1, num=target_count, dtype=np.int64)
    return array[indices].astype(np.float32, copy=False)


def normalise_placement_candidates(
    raw_candidates: Any,
    *,
    expected_count: int | None = DEFAULT_CANDIDATE_LIMIT,
) -> list[dict[str, Any]]:
    np, _Image = _load_numeric_deps()
    try:
        arr = np.asarray(raw_candidates, dtype=np.float64)
    except Exception as exc:  # noqa: BLE001
        raise AnyPlaceInputError("inconsistent_placement_outputs") from exc
    if arr.size == 0:
        raise AnyPlaceInputError("no_placement_candidates")
    if arr.ndim != 3 or arr.shape[1:] != (4, 4):
        raise AnyPlaceInputError("inconsistent_placement_outputs")
    if expected_count is not None and arr.shape[0] != expected_count:
        raise AnyPlaceInputError("inconsistent_placement_outputs")
    if not np.isfinite(arr).all():
        raise AnyPlaceInputError("inconsistent_placement_outputs")
    candidates = []
    for idx, pose in enumerate(arr):
        if not _is_rigid_transform(pose):
            raise AnyPlaceInputError("inconsistent_placement_outputs")
        candidates.append(
            {
                "id": f"placement_{idx:03d}",
                "object_placement_transform": {
                    "frame": FRAME,
                    "camera_frame": CAMERA_FRAME,
                    "convention": POSE_CONVENTION,
                    "transform_matrix": _float_matrix(pose),
                },
            }
        )
    return candidates


def _model_world_to_placement_camera_transforms(
    raw_candidates: Any,
    *,
    placement_camera_to_world: Any,
    np: Any,
) -> Any:
    """Convert official world-frame transforms back to the public camera frame."""

    transforms = np.asarray(raw_candidates, dtype=np.float64)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        # Leave shape validation and its public error reason to the normalizer.
        return raw_candidates
    inverse = np.linalg.inv(placement_camera_to_world)
    return np.matmul(
        np.matmul(inverse[None, :, :], transforms),
        placement_camera_to_world[None, :, :],
    )


def _project_object_bottoms_to_support(
    raw_candidates: Any,
    *,
    object_points: Any,
    support_points: Any,
    np: Any,
) -> Any:
    """Place each predicted object bottom on the measured gravity-aligned support."""

    transforms = np.asarray(raw_candidates, dtype=np.float64)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        return raw_candidates
    object_array = np.asarray(object_points, dtype=np.float64)
    support_array = np.asarray(support_points, dtype=np.float64)
    if (
        object_array.ndim != 2
        or object_array.shape[1] != 3
        or len(object_array) == 0
        or support_array.ndim != 2
        or support_array.shape[1] != 3
        or len(support_array) == 0
    ):
        return raw_candidates
    # Median is robust to the marker border/depth holes; use a low percentile
    # for the object bottom so a few noisy depth pixels cannot suspend it.
    support_height = float(np.median(support_array[:, 2]))
    homogeneous = np.concatenate(
        [object_array, np.ones((len(object_array), 1), dtype=np.float64)], axis=1
    )
    projected = np.matmul(transforms, homogeneous.T).transpose(0, 2, 1)[:, :, :3]
    bottoms = np.quantile(projected[:, :, 2], 0.02, axis=1)
    adjusted = transforms.copy()
    adjusted[:, 2, 3] += support_height - bottoms
    return adjusted


def _is_rigid_transform(matrix: Any) -> bool:
    np, _Image = _load_numeric_deps()
    return bool(
        matrix.shape == (4, 4)
        and np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
        and _is_rotation_matrix(matrix[:3, :3])
    )


def _is_rotation_matrix(rotation: Any) -> bool:
    np, _Image = _load_numeric_deps()
    return bool(
        rotation.shape == (3, 3)
        and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        and np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
    )


def _decode_image_payload(
    payload: dict[str, Any] | None,
    *,
    Image: Any,
    np: Any,
    convert: str | None,
    missing_reason: str,
    decode_reason: str,
) -> Any:
    if not isinstance(payload, dict) or not payload.get("base64"):
        raise AnyPlaceInputError(missing_reason)
    try:
        data = base64.b64decode(payload["base64"], validate=True)
        image = Image.open(io.BytesIO(data))
        if convert:
            image = image.convert(convert)
        return np.asarray(image)
    except Exception as exc:  # noqa: BLE001
        raise AnyPlaceInputError(decode_reason) from exc


def _load_numeric_deps() -> tuple[Any, Any]:
    import numpy as np
    from PIL import Image

    return np, Image


def _float_list(values: Any) -> list[float]:
    return [float(value) for value in values]


def _float_matrix(values: Any) -> list[list[float]]:
    return [_float_list(row) for row in values]


def _metadata_base() -> dict[str, Any]:
    return {
        "pose_convention": POSE_CONVENTION,
        "candidate_limit": DEFAULT_CANDIDATE_LIMIT,
        "model_sample_count": MODEL_SAMPLE_COUNT,
    }


def _with_duration(metadata: dict[str, Any], start: float) -> dict[str, Any]:
    return {**metadata, "duration_s": round(time.perf_counter() - start, 4)}


def _failure_result(
    *,
    reason: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "success": False,
        "content": content,
        "details": {
            "tool": "anyplace",
            "backend": "anyplace_mcp",
            "model": MODEL_NAME,
            "frame": FRAME,
            "camera_frame": CAMERA_FRAME,
            "candidate_count": 0,
            "placement_candidates": [],
            "reason": reason,
            "metadata": dict(metadata or {}),
        },
    }
