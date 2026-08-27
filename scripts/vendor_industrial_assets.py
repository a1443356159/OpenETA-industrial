#!/usr/bin/env python3
"""Build the pinned, offline industrial-workcell visual asset closure.

This script is intentionally dependency free.  It converts the binary PLY
files from the STARS Workshop Tools Dataset into Gazebo-friendly OBJ/MTL
meshes, normalizes every tool to metres and a common tabletop convention, and
builds deterministic metric-bolt meshes with visible helical threads.

The source repositories and exact input hashes are recorded in the adjacent
``asset_manifest.json``.  Generated files are committed so production and
acceptance runs never download assets at launch time.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import struct
from typing import Iterable, Sequence


TOOL_SPECS = {
    "adjustable_wrench": ("Adjustable Wrench", (0, 2, 1), 0.220),
    "box_wrench": ("Box Wrench", (0, 1, 2), 0.190),
    "pliers": ("Pliers", (0, 1, 2), 0.200),
    "screwdriver": ("Screwdriver", (2, 0, 1), 0.180),
    "allen_key": ("Allen Key", (1, 0, 2), 0.160),
}

TOOL_PALETTES = {
    "adjustable_wrench": {
        (165, 158, 150, 0): (224, 157, 24, 255),
        (165, 132, 0, 0): (92, 70, 24, 255),
    },
    "box_wrench": {
        (135, 140, 140, 0): (194, 201, 211, 255),
        (64, 0, 64, 0): (88, 96, 106, 255),
    },
    "pliers": {
        (229, 229, 229, 0): (176, 184, 196, 255),
        (0, 255, 255, 0): (26, 83, 192, 255),
    },
    "screwdriver": {
        (2, 61, 210, 0): (28, 48, 84, 255),
        (229, 229, 229, 0): (188, 196, 208, 255),
    },
    "allen_key": {(198, 198, 208, 0): (194, 201, 211, 255)},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_ply(path: Path) -> tuple[list[tuple[float, float, float]], list[tuple[tuple[int, int, int, int], tuple[int, ...]]]]:
    data = path.read_bytes()
    marker = b"end_header\r\n"
    header_end = data.find(marker)
    if header_end < 0:
        marker = b"end_header\n"
        header_end = data.find(marker)
    if header_end < 0:
        raise RuntimeError(f"PLY header is incomplete: {path}")
    payload_offset = header_end + len(marker)
    header = data[:payload_offset].decode("ascii")
    if "format binary_little_endian 1.0" not in header:
        raise RuntimeError(f"unsupported PLY encoding: {path}")
    counts = {
        parts[1]: int(parts[2])
        for line in header.splitlines()
        if (parts := line.split())[:1] == ["element"] and len(parts) == 3
    }
    vertex_count = counts.get("vertex", 0)
    face_count = counts.get("face", 0)
    if vertex_count <= 0 or face_count <= 0:
        raise RuntimeError(f"PLY topology is empty: {path}")
    vertices = [
        struct.unpack_from("<fff", data, payload_offset + index * 12)
        for index in range(vertex_count)
    ]
    offset = payload_offset + vertex_count * 12
    faces: list[tuple[tuple[int, int, int, int], tuple[int, ...]]] = []
    for _ in range(face_count):
        rgba = struct.unpack_from("<BBBB", data, offset)
        offset += 4
        index_count = data[offset]
        offset += 1
        if index_count < 3:
            raise RuntimeError(f"PLY face is degenerate: {path}")
        indices = struct.unpack_from(f"<{index_count}i", data, offset)
        offset += index_count * 4
        faces.append((rgba, indices))
    if offset != len(data):
        raise RuntimeError(f"PLY payload has unexpected trailing data: {path}")
    return vertices, faces


def _normalize_vertices(
    vertices: Sequence[Sequence[float]],
    *,
    axes: tuple[int, int, int],
    target_length_m: float,
) -> list[tuple[float, float, float]]:
    remapped = [tuple(float(vertex[axis]) for axis in axes) for vertex in vertices]
    minimum = [min(vertex[axis] for vertex in remapped) for axis in range(3)]
    maximum = [max(vertex[axis] for vertex in remapped) for axis in range(3)]
    scale = target_length_m / (maximum[0] - minimum[0])
    center_x = (minimum[0] + maximum[0]) / 2.0
    center_y = (minimum[1] + maximum[1]) / 2.0
    return [
        (
            (vertex[0] - center_x) * scale,
            (vertex[1] - center_y) * scale,
            (vertex[2] - minimum[2]) * scale,
        )
        for vertex in remapped
    ]


def _write_colored_obj(
    output: Path,
    *,
    vertices: Sequence[Sequence[float]],
    faces: Iterable[tuple[tuple[int, int, int, int], Sequence[int]]],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[int, int, int, int], list[Sequence[int]]] = defaultdict(list)
    for color, indices in faces:
        groups[color].append(indices)
    material_path = output.with_suffix(".mtl")
    with material_path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("# Deterministically converted STARS face materials.\n")
        for index, color in enumerate(sorted(groups)):
            r, g, b, a = (channel / 255.0 for channel in color)
            stream.write(f"newmtl face_{index:03d}\n")
            stream.write(f"Ka {r:.6f} {g:.6f} {b:.6f}\n")
            stream.write(f"Kd {r:.6f} {g:.6f} {b:.6f}\n")
            stream.write("Ks 0.180000 0.180000 0.180000\n")
            stream.write("Ns 48.000000\n")
            stream.write(f"d {a:.6f}\n\n")
    with output.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("# OpenETA offline industrial visual mesh; units: metres.\n")
        stream.write(f"mtllib {material_path.name}\n")
        for x, y, z in vertices:
            stream.write(f"v {x:.9f} {y:.9f} {z:.9f}\n")
        stream.write("s 1\n")
        for material_index, color in enumerate(sorted(groups)):
            stream.write(f"usemtl face_{material_index:03d}\n")
            for indices in groups[color]:
                stream.write("f " + " ".join(str(index + 1) for index in indices) + "\n")


def _write_glb(
    output: Path,
    *,
    vertices: Sequence[Sequence[float]],
    faces: Iterable[tuple[str, Sequence[int]]],
    colors: dict[str, tuple[float, float, float, float]],
) -> None:
    """Write a compact GLB with explicit PBR materials and smooth normals."""

    import math

    grouped: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    normals = [[0.0, 0.0, 0.0] for _ in vertices]
    for material, polygon in faces:
        for offset in range(1, len(polygon) - 1):
            triangle = (int(polygon[0]), int(polygon[offset]), int(polygon[offset + 1]))
            grouped[material].append(triangle)
            a, b, c = (vertices[index] for index in triangle)
            ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
            ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
            normal = (
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            )
            for index in triangle:
                for axis in range(3):
                    normals[index][axis] += normal[axis]
    normalized_normals = []
    for normal in normals:
        length = math.sqrt(sum(value * value for value in normal))
        normalized_normals.append(
            tuple(value / length for value in normal) if length > 1e-12 else (0.0, 0.0, 1.0)
        )

    binary = bytearray()
    views: list[dict[str, object]] = []
    accessors: list[dict[str, object]] = []

    def append_view(payload: bytes, *, target: int) -> int:
        while len(binary) % 4:
            binary.append(0)
        offset = len(binary)
        binary.extend(payload)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(payload), "target": target})
        return len(views) - 1

    position_payload = b"".join(struct.pack("<fff", *vertex) for vertex in vertices)
    position_view = append_view(position_payload, target=34962)
    minimum = [min(float(vertex[axis]) for vertex in vertices) for axis in range(3)]
    maximum = [max(float(vertex[axis]) for vertex in vertices) for axis in range(3)]
    accessors.append({"bufferView": position_view, "componentType": 5126, "count": len(vertices), "type": "VEC3", "min": minimum, "max": maximum})
    normal_view = append_view(
        b"".join(struct.pack("<fff", *normal) for normal in normalized_normals),
        target=34962,
    )
    accessors.append({"bufferView": normal_view, "componentType": 5126, "count": len(vertices), "type": "VEC3"})

    material_names = sorted(grouped)
    primitives = []
    for material_index, material in enumerate(material_names):
        flat_indices = [index for triangle in grouped[material] for index in triangle]
        index_view = append_view(
            b"".join(struct.pack("<I", index) for index in flat_indices),
            target=34963,
        )
        accessor_index = len(accessors)
        accessors.append({"bufferView": index_view, "componentType": 5125, "count": len(flat_indices), "type": "SCALAR", "min": [min(flat_indices)], "max": [max(flat_indices)]})
        primitives.append({"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": accessor_index, "material": material_index, "mode": 4})
    materials = []
    for name in material_names:
        rgba = colors[name]
        materials.append({
            "name": name,
            "doubleSided": True,
            "pbrMetallicRoughness": {
                "baseColorFactor": list(rgba),
                "metallicFactor": 0.18,
                "roughnessFactor": 0.43,
            },
        })
    document = {
        "asset": {"version": "2.0", "generator": "OpenETA deterministic industrial asset builder"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": primitives}],
        "materials": materials,
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(binary)}],
    }
    json_payload = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_payload += b" " * ((-len(json_payload)) % 4)
    binary.extend(b"\0" * ((-len(binary)) % 4))
    total_length = 12 + 8 + len(json_payload) + 8 + len(binary)
    output.write_bytes(
        struct.pack("<III", 0x46546C67, 2, total_length)
        + struct.pack("<II", len(json_payload), 0x4E4F534A)
        + json_payload
        + struct.pack("<II", len(binary), 0x004E4942)
        + binary
    )


def _append_quad(faces: list[tuple[str, tuple[int, ...]]], material: str, a: int, b: int, c: int, d: int) -> None:
    faces.append((material, (a, b, c, d)))


def _bolt_mesh(
    *,
    nominal_diameter_m: float,
    shaft_length_m: float,
    pitch_m: float,
    head_kind: str,
) -> tuple[list[tuple[float, float, float]], list[tuple[str, tuple[int, ...]]]]:
    """Return a tabletop-oriented bolt with an explicit helical thread."""

    import math

    radial_segments = 24
    # Eight axial samples per pitch resolve the triangular thread profile;
    # using one full radial ring per angular segment triples file size without
    # a visible benefit at acceptance-camera distance.
    axial_steps = max(24, round(shaft_length_m / pitch_m * 8))
    major_radius = nominal_diameter_m / 2.0
    minor_radius = major_radius * 0.82
    shaft_start = 0.0
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[str, tuple[int, ...]]] = []
    for axial in range(axial_steps + 1):
        x = shaft_start + shaft_length_m * axial / axial_steps
        for radial in range(radial_segments):
            theta = 2.0 * math.pi * radial / radial_segments
            thread_phase = (theta - 2.0 * math.pi * x / pitch_m) / (2.0 * math.pi)
            fraction = thread_phase - math.floor(thread_phase)
            tooth = 1.0 - abs(2.0 * fraction - 1.0)
            radius = minor_radius + (major_radius - minor_radius) * tooth
            vertices.append((x, radius * math.cos(theta), radius * math.sin(theta)))
    for axial in range(axial_steps):
        for radial in range(radial_segments):
            next_radial = (radial + 1) % radial_segments
            a = axial * radial_segments + radial
            b = axial * radial_segments + next_radial
            c = (axial + 1) * radial_segments + next_radial
            d = (axial + 1) * radial_segments + radial
            _append_quad(faces, "thread", a, b, c, d)

    head_length = nominal_diameter_m * (0.78 if head_kind == "hex" else 0.90)
    head_radius = nominal_diameter_m * (0.90 if head_kind == "hex" else 0.82)
    sides = 6 if head_kind == "hex" else 32
    head_start = -head_length
    base = len(vertices)
    for x in (head_start, 0.0):
        for side in range(sides):
            theta = 2.0 * math.pi * side / sides + (math.pi / 6.0 if sides == 6 else 0.0)
            vertices.append((x, head_radius * math.cos(theta), head_radius * math.sin(theta)))
    for side in range(sides):
        nxt = (side + 1) % sides
        _append_quad(faces, "head", base + side, base + nxt, base + sides + nxt, base + sides + side)
    front_center = len(vertices)
    vertices.append((head_start, 0.0, 0.0))
    for side in range(sides):
        nxt = (side + 1) % sides
        faces.append(("head", (front_center, base + nxt, base + side)))

    # A dark inset distinguishes the socket-head fastener without relying on
    # texture transport.  It sits 0.2 mm proud to avoid z-fighting.
    if head_kind == "socket":
        socket_base = len(vertices)
        socket_radius = nominal_diameter_m * 0.38
        for side in range(6):
            theta = 2.0 * math.pi * side / 6.0
            vertices.append((head_start - 0.0002, socket_radius * math.cos(theta), socket_radius * math.sin(theta)))
        socket_center = len(vertices)
        vertices.append((head_start - 0.0002, 0.0, 0.0))
        for side in range(6):
            nxt = (side + 1) % 6
            faces.append(("socket", (socket_center, socket_base + side, socket_base + nxt)))

    minimum_y = min(vertex[1] for vertex in vertices)
    # Rotate the radial cross-section so the bolt rests on z=0 and center the
    # full length around x=0 for intuitive scene placement.
    center_x = (head_start + shaft_length_m) / 2.0
    tabletop = [(x - center_x, z, y - minimum_y) for x, y, z in vertices]
    return tabletop, faces


def _write_bolt_obj(output: Path, *, color: tuple[float, float, float], **kwargs: object) -> None:
    vertices, faces = _bolt_mesh(**kwargs)
    output.parent.mkdir(parents=True, exist_ok=True)
    material = output.with_suffix(".mtl")
    with material.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("newmtl thread\n")
        stream.write(f"Ka {color[0] * 0.55:.6f} {color[1] * 0.55:.6f} {color[2] * 0.55:.6f}\n")
        stream.write(f"Kd {color[0]:.6f} {color[1]:.6f} {color[2]:.6f}\nKs 0.42 0.42 0.42\nNs 72\n\n")
        stream.write("newmtl head\n")
        stream.write(f"Ka {color[0] * 0.65:.6f} {color[1] * 0.65:.6f} {color[2] * 0.65:.6f}\n")
        stream.write(f"Kd {min(color[0] * 1.08, 1):.6f} {min(color[1] * 1.08, 1):.6f} {min(color[2] * 1.08, 1):.6f}\nKs 0.55 0.55 0.55\nNs 96\n\n")
        stream.write("newmtl socket\nKa 0.01 0.01 0.012\nKd 0.025 0.025 0.03\nKs 0.08 0.08 0.08\nNs 24\n")
    with output.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(f"mtllib {material.name}\n")
        for x, y, z in vertices:
            stream.write(f"v {x:.9f} {y:.9f} {z:.9f}\n")
        stream.write("s 1\n")
        active = ""
        for face_material, indices in faces:
            if face_material != active:
                active = face_material
                stream.write(f"usemtl {active}\n")
            stream.write("f " + " ".join(str(index + 1) for index in indices) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workshop-root", type=Path, required=True)
    parser.add_argument("--picking-bin-glb", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    source_hashes: dict[str, str] = {}
    for asset_id, (source_name, axes, target_length) in TOOL_SPECS.items():
        source = args.workshop_root / source_name / "mesh.ply"
        vertices, faces = _read_ply(source)
        normalized = _normalize_vertices(vertices, axes=axes, target_length_m=target_length)
        palette = TOOL_PALETTES[asset_id]
        recolored = [(palette.get(color, (*color[:3], 255)), indices) for color, indices in faces]
        _write_colored_obj(
            output / f"{asset_id}.obj",
            vertices=normalized,
            faces=recolored,
        )
        material_colors = {
            f"surface_{index:03d}": tuple(channel / 255.0 for channel in color)
            for index, color in enumerate(sorted({color for color, _ in recolored}))
        }
        color_names = {
            color: f"surface_{index:03d}"
            for index, color in enumerate(sorted({color for color, _ in recolored}))
        }
        _write_glb(
            output / f"{asset_id}.glb",
            vertices=normalized,
            faces=((color_names[color], indices) for color, indices in recolored),
            colors=material_colors,
        )
        source_hashes[f"stars:{source_name}/mesh.ply"] = _sha256(source)

    _write_bolt_obj(
        output / "hex_bolt_24mm.obj",
        color=(0.82, 0.035, 0.025),
        nominal_diameter_m=0.024,
        shaft_length_m=0.095,
        pitch_m=0.003,
        head_kind="hex",
    )
    bolt_vertices, bolt_faces = _bolt_mesh(
        nominal_diameter_m=0.024,
        shaft_length_m=0.095,
        pitch_m=0.003,
        head_kind="hex",
    )
    _write_glb(
        output / "hex_bolt_24mm.glb",
        vertices=bolt_vertices,
        faces=bolt_faces,
        colors={"thread": (0.72, 0.025, 0.018, 1.0), "head": (0.90, 0.04, 0.025, 1.0)},
    )
    _write_bolt_obj(
        output / "socket_bolt_20mm.obj",
        color=(0.95, 0.24, 0.025),
        nominal_diameter_m=0.020,
        shaft_length_m=0.080,
        pitch_m=0.0025,
        head_kind="socket",
    )
    bolt_vertices, bolt_faces = _bolt_mesh(
        nominal_diameter_m=0.020,
        shaft_length_m=0.080,
        pitch_m=0.0025,
        head_kind="socket",
    )
    _write_glb(
        output / "socket_bolt_20mm.glb",
        vertices=bolt_vertices,
        faces=bolt_faces,
        colors={"thread": (0.82, 0.16, 0.012, 1.0), "head": (0.98, 0.28, 0.02, 1.0), "socket": (0.018, 0.020, 0.025, 1.0)},
    )
    shutil.copyfile(args.picking_bin_glb, output / "picking_bin.glb")
    source_hashes["fuel:Picking_Bin/meshes/model.glb"] = _sha256(args.picking_bin_glb)
    generated = {
        path.name: _sha256(path)
        for path in sorted(output.iterdir())
        if path.suffix in {".obj", ".mtl", ".glb"}
    }
    print(json.dumps({"source_sha256": source_hashes, "generated_sha256": generated}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
