import torch
import SMPLicit
import trimesh
import numpy as np

SMPLicit_layer = SMPLicit.SMPLicit()
meshes = SMPLicit_layer.reconstruct()

verts = np.concatenate((meshes[0].vertices, meshes[1].vertices))
faces = np.concatenate((meshes[0].faces, meshes[1].faces + len(meshes[0].vertices)))
mesh = trimesh.Trimesh(verts, faces)
mesh.show()
