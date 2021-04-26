import torch
import SMPLicit
import trimesh
import numpy as np
import os

SMPLicit_layer = SMPLicit.SMPLicit()

iterations = 10

os.mkdir('../interpolation')
cool_latent_reps = np.load('z_gaussians.npy')

params_start = cool_latent_reps[0]
params_end = cool_latent_reps[-1]

for i in range(iterations + 1):
    params = (params_start*(iterations - i) + params_end*i)/iterations

    meshes = SMPLicit_layer.reconstruct(Zs=[params])
    verts = np.concatenate((meshes[0].vertices, meshes[1].vertices))
    faces = np.concatenate((meshes[0].faces, meshes[1].faces + len(meshes[0].vertices)))

    mesh = trimesh.Trimesh(verts, faces)
    mesh.export('../interpolation/mesh_%d.obj'%i)
