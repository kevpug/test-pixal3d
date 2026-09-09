"""
CPU mesh operations standing in for ``cumesh``.

``cumesh`` does hole filling, quadric simplification, topology repair and UV
unwrapping on the GPU. Its CUDA sources go through hipify cleanly enough on
Linux, but there is no practical way to build it on Windows against the ROCm
SDK, so this module reaches for the established CPU equivalents instead:
``trimesh`` for repair, ``fast_simplification`` for quadric decimation, and
``xatlas`` for the atlas — the same unwrapper ``cumesh`` itself vendors.

These run on the CPU and are the slowest part of a fallback run; the decimation
target is what controls that cost, so it is worth turning down on a laptop.
"""

from typing import *
import numpy as np


__all__ = [
    'clean_mesh',
    'simplify_mesh',
    'uv_unwrap',
    'vertex_normals',
    'missing_dependencies',
]


def missing_dependencies() -> List[str]:
    """Names of the CPU packages this module needs but cannot import."""
    missing = []
    for mod, pkg in (('trimesh', 'trimesh'),
                     ('xatlas', 'xatlas'),
                     ('fast_simplification', 'fast-simplification')):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    return missing


def _trimesh(vertices: np.ndarray, faces: np.ndarray):
    import trimesh
    return trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
        validate=False,
    )


def clean_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    min_component_area: float = 1e-5,
    fill_holes: bool = True,
    unify_orientation: bool = False,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Drop degenerate and duplicate faces, remove specks, and optionally fill
    holes and unify winding — the ``cumesh`` cleanup sequence, on the CPU.
    """
    import trimesh

    mesh = _trimesh(vertices, faces)
    if len(mesh.faces) == 0:
        return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)

    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())

    if min_component_area > 0 and len(mesh.faces) > 0:
        components = trimesh.graph.connected_components(
            mesh.face_adjacency, min_len=1, nodes=np.arange(len(mesh.faces)))
        if len(components) > 1:
            areas = mesh.area_faces
            keep = [c for c in components if float(areas[c].sum()) >= min_component_area]
            if keep and len(keep) != len(components):
                mask = np.zeros(len(mesh.faces), dtype=bool)
                mask[np.concatenate(keep)] = True
                mesh.update_faces(mask)
                if verbose:
                    print(f"  dropped {len(components) - len(keep)} small components")

    mesh.remove_unreferenced_vertices()

    if unify_orientation and len(mesh.faces) > 0:
        try:
            trimesh.repair.fix_winding(mesh)
            trimesh.repair.fix_inversion(mesh)
        except Exception as exc:            # non-manifold input; keep what we have
            if verbose:
                print(f"  orientation unification skipped: {exc}")

    if fill_holes and len(mesh.faces) > 0:
        try:
            mesh.fill_holes()
        except Exception as exc:
            if verbose:
                print(f"  hole filling skipped: {exc}")

    return (np.ascontiguousarray(mesh.vertices, dtype=np.float32),
            np.ascontiguousarray(mesh.faces, dtype=np.int32))


def simplify_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: int,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Quadric-decimate to ``target_faces``; a no-op if already below it."""
    import fast_simplification

    faces = np.asarray(faces, dtype=np.int32)
    if target_faces <= 0 or len(faces) <= target_faces:
        return np.asarray(vertices, dtype=np.float32), faces

    out_v, out_f = fast_simplification.simplify(
        np.asarray(vertices, dtype=np.float32),
        faces,
        target_count=int(target_faces),
    )
    if verbose:
        print(f"  simplified {len(faces)} -> {len(out_f)} faces")
    return (np.ascontiguousarray(out_v, dtype=np.float32),
            np.ascontiguousarray(out_f, dtype=np.int32))


def uv_unwrap(
    vertices: np.ndarray,
    faces: np.ndarray,
    texture_size: int = 2048,
    padding: int = 4,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Cut and pack a UV atlas.

    Returns ``(vertices, faces, uvs, vmapping)`` where ``vmapping`` indexes the
    input vertices — unwrapping duplicates vertices along seams, and the caller
    needs that map to carry per-vertex data (normals) across.
    """
    import xatlas

    atlas = xatlas.Atlas()
    atlas.add_mesh(
        np.asarray(vertices, dtype=np.float32),
        np.asarray(faces, dtype=np.uint32),
    )
    pack_options = xatlas.PackOptions()
    pack_options.resolution = int(texture_size)
    pack_options.padding = int(padding)
    pack_options.bruteForce = False
    chart_options = xatlas.ChartOptions()
    atlas.generate(chart_options=chart_options, pack_options=pack_options)

    vmapping, indices, uvs = atlas[0]
    if verbose:
        print(f"  atlas {atlas.width}x{atlas.height}, {len(vmapping)} verts, {len(indices)} faces")

    vmapping = np.asarray(vmapping, dtype=np.int64)
    return (np.ascontiguousarray(np.asarray(vertices, dtype=np.float32)[vmapping]),
            np.ascontiguousarray(indices, dtype=np.int32),
            np.ascontiguousarray(uvs, dtype=np.float32),
            vmapping)


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals."""
    mesh = _trimesh(vertices, faces)
    return np.ascontiguousarray(mesh.vertex_normals, dtype=np.float32)
