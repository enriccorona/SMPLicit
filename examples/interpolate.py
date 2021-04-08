import torch
import SMPLicit
import trimesh
import numpy as np

SMPLicit_layer = SMPLicit.SMPLicit()

iterations = 10

cool_latent_reps = np.load('/home/enric/cvpr21/model/checkpoints/upperbody_combine_22sep_size126_biggershape_1_002_LRimageencoderX10_rate0.5_noisyZ100_withbetaaugmentation_11epochs/z_gaussians.npy')

params_start = cool_latent_reps[0]
params_end = cool_latent_reps[-1]

for i in range(iterations + 1):
    params = (params_start*(iterations - i) + params_end*i)/iterations

    meshes = SMPLicit_layer.reconstruct(Zs=[params])
    verts = np.concatenate((meshes[0].vertices, meshes[1].vertices))
    faces = np.concatenate((meshes[0].faces, meshes[1].faces + len(meshes[0].vertices)))

    mesh = trimesh.Trimesh(verts, faces)
    mesh.export('../interpolation/mesh_%d.obj'%i)
