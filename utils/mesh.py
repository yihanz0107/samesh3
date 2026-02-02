import numpy as np
import trimesh
from trimesh.base import Trimesh, Scene

from data.common import NumpyTensor, TorchTensor


def duplicate_verts(mesh: Trimesh) -> Trimesh:
    """
    Call before coloring mesh to avoid face interpolation since openGL stores color attributes per vertex.

        ...
        mesh = duplicate_verts(mesh)
        mesh.visual.face_colors = colors
        ...

    NOTE: removes visuals for verticies, but preserves for faces.
    """
    verts = mesh.vertices[mesh.faces.reshape(-1), :]
    faces = np.arange(0, verts.shape[0])
    faces = faces.reshape(-1, 3)
    return Trimesh(vertices=verts, faces=faces, face_colors=mesh.visual.face_colors, process=False)


def handle_pose(pose: NumpyTensor['4 4']) -> NumpyTensor['4 4']:
    """
    Handles common case that results in numerical instability in rendering faceids:

        ...
        pose, _ = scene.graph[name]
        pose = handle_pose(pose)
        ...
    """
    identity = np.eye(4)
    if np.allclose(pose, identity, atol=1e-6):
        return identity
    return pose


def transform(pose: NumpyTensor['4 4'], vertices: NumpyTensor['nv 3']) -> NumpyTensor['nv 3']:
    """
    """
    homogeneous = np.concatenate([vertices, np.ones((vertices.shape[0], 1))], axis=1)
    return (pose @ homogeneous.T).T[:, :3]


def concat_scene_vertices(scene: Scene) -> NumpyTensor['nv 3']:
    """
    """
    verts = []
    for name, geom in scene.geometry.items():
        if name in scene.graph:
            pose, _ = scene.graph[name]
            pose = handle_pose(pose)
            geom.vertices = transform(pose, geom.vertices)
        verts.append(geom.vertices)
    return np.concatenate(verts)


def bounding_box(vertices: NumpyTensor['n 3']) -> NumpyTensor['2 3']:
    """
    Compute bounding box from vertices.
    """
    return np.array([vertices.min(axis=0), vertices.max(axis=0)])


def bounding_box_centroid(vertices: NumpyTensor['n 3']) -> NumpyTensor['3']:
    """
    Compute bounding box centroid from vertices.
    """
    return bounding_box(vertices).mean(axis=0)


def norm_mesh(mesh: Trimesh) -> Trimesh:
    """
    Normalize mesh vertices to bounding box [-1, 1]. 
    
    NOTE:: In place operation that consumes mesh.
    """
    centroid = bounding_box_centroid(mesh.vertices)
    mesh.vertices -= centroid
    mesh.vertices /= np.abs(mesh.vertices).max()
    mesh.vertices *= (1 - 1e-3)
    return mesh


def norm_scene(scene: Scene) -> Scene:
    """
    Normalize scene vertices to bounding box [-1, 1]. 
    
    NOTE:: In place operation that consumes scene.
    """
    centroid = bounding_box_centroid(concat_scene_vertices(scene))
    for geom in scene.geometry.values():
        geom.vertices -= centroid
    extent = np.abs(concat_scene_vertices(scene)).max()
    for geom in scene.geometry.values():
        geom.vertices /= extent
        geom.vertices *= (1 - 1e-3)
    return scene
