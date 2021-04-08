import torch
import SMPLicit
import trimesh
import numpy as np

SMPLicit_layer = SMPLicit.SMPLicit()

upperbody_Z = np.zeros(18)
pants_Z = np.zeros(18)
hair_Z = np.zeros(18)
shoes_Z = np.zeros(4)
Zs = [upperbody_Z, pants_Z, hair_Z, shoes_Z]
meshes = SMPLicit_layer.reconstruct(model_ids=[0, 1, 3, 4], Zs=Zs)

verts = np.zeros((0, 3))
faces = np.zeros((0, 3))
for mesh in meshes:
    faces = np.concatenate((faces, mesh.faces + len(verts)))
    verts = np.concatenate((verts, mesh.vertices))

mesh = trimesh.Trimesh(verts, faces)
mesh.show()
