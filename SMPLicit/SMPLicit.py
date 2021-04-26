import torch
import numpy as np
import torch.nn as nn
import os
import trimesh
from .SMPL import SMPL
import kaolin
from .SMPLicit_options import Options
from .smplicit_core_test import Model

class SMPLicit(nn.Module):
    def __init__(self):
        super(SMPLicit, self).__init__()
        _opt = Options().parse()

        uppercloth = Model(_opt.path_checkpoints + '/upperclothes.pth',
                                        _opt.upperbody_n_z_cut, 
                                        _opt.upperbody_n_z_style, _opt.upperbody_num_clusters, 
                                        _opt.path_cluster_files + _opt.upperbody_clusters, 
                                        _opt.upperbody_b_min, _opt.upperbody_b_max,
                                        _opt.upperbody_resolution, thresh=_opt.upperbody_thresh_occupancy)

        pants = Model(_opt.path_checkpoints + '/pants.pth',
                                        _opt.pants_n_z_cut, 
                                        _opt.pants_n_z_style, _opt.pants_num_clusters, 
                                        _opt.path_cluster_files + _opt.pants_clusters, 
                                        _opt.pants_b_min, _opt.pants_b_max,
                                        _opt.pants_resolution, thresh=_opt.pants_thresh_occupancy)

        skirts = Model(_opt.path_checkpoints + '/skirts.pth',
                                        _opt.skirts_n_z_cut, 
                                        _opt.skirts_n_z_style, _opt.skirts_num_clusters, 
                                        _opt.path_cluster_files + _opt.skirts_clusters, 
                                        _opt.skirts_b_min, _opt.skirts_b_max,
                                        _opt.skirts_resolution, thresh=_opt.skirts_thresh_occupancy)

        hair = Model(_opt.path_checkpoints + '/hair.pth',
                                        _opt.hair_n_z_cut, 
                                        _opt.hair_n_z_style, _opt.hair_num_clusters, 
                                        _opt.path_cluster_files + _opt.hair_clusters, 
                                        _opt.hair_b_min, _opt.hair_b_max,
                                        _opt.hair_resolution, thresh=_opt.hair_thresh_occupancy)

        shoes = Model(_opt.path_checkpoints + '/shoes.pth',
                                        _opt.shoes_n_z_cut, 
                                        _opt.shoes_n_z_style, _opt.shoes_num_clusters, 
                                        _opt.path_cluster_files + _opt.shoes_clusters, 
                                        _opt.shoes_b_min, _opt.shoes_b_max,
                                        _opt.shoes_resolution, thresh=_opt.shoes_thresh_occupancy)

        self.models = [uppercloth, pants, skirts, hair, shoes]

        self.SMPL_Layer = SMPL(_opt.path_SMPL, obj_saveable=True).cuda()
        self.smpl_faces = self.SMPL_Layer.faces

        self.Astar_pose = torch.zeros(1, 72).cuda()
        self.Astar_pose[0, 5] = 0.04
        self.Astar_pose[0, 8] = -0.04

        self._opt = _opt

        # HYPERPARAMETER: Maximum number of points used when reposing.
        # This takes a lot of memory when finding the closest point in the SMPL so doing it by steps
        self.step = 1000 

    def reconstruct(self, model_ids=[0], Zs=[np.zeros(18)], pose=np.zeros(72), beta=np.zeros(10)):
        # Prepare tensors:
        for i in range(len(Zs)):
            if not torch.is_tensor(Zs[i]):
                Zs[i] = torch.FloatTensor(Zs[i])
            Zs[i] = Zs[i].cuda()
        if not torch.is_tensor(pose):
            pose = torch.FloatTensor(pose)
        if not torch.is_tensor(beta):
            beta = torch.FloatTensor(beta)

        pose = pose.reshape(1,-1).cuda()
        beta = beta.reshape(1,-1).cuda()

        posed_smpl = self.SMPL_Layer.forward(beta=beta, theta=pose, get_skin=True)[0][0].cpu().data.numpy()
        J, unposed_smpl = self.SMPL_Layer.skeleton(beta, require_body=True)
        Astar_smpl = self.SMPL_Layer.forward(beta=beta, theta=self.Astar_pose, get_skin=True)[0][0]
        inference_mesh = kaolin.rep.TriangleMesh.from_tensors(unposed_smpl[0], 
                                        torch.LongTensor(self.smpl_faces).cuda())
        inference_lowerbody = kaolin.rep.TriangleMesh.from_tensors(Astar_smpl,
                                        torch.LongTensor(self.smpl_faces).cuda())

        out_meshes = [trimesh.Trimesh(posed_smpl, self.smpl_faces, process=False)]
        for i, id_ in enumerate(model_ids):
            if id_ == 1 or id_ == 2:
                mesh = self.models[id_].reconstruct(Zs[i].cpu().data.numpy(), inference_lowerbody)
                mesh = self.pose_mesh_lowerbody(mesh, pose, beta, J, unposed_smpl)
            else:
                mesh = self.models[id_].reconstruct(Zs[i].cpu().data.numpy(), inference_mesh)
                if id_ == 4: # Shoe: Duplicate left shoe
                    mesh = self.get_right_shoe(mesh)
                mesh = self.pose_mesh(mesh, pose, J, unposed_smpl)

            out_meshes.append(mesh)

        return out_meshes

    def get_right_shoe(self, mesh):
        right_vshoe = mesh.vertices.copy()
        right_vshoe[:, 0] *= -1
        fshoe = mesh.faces

        mesh = trimesh.Trimesh(np.concatenate((mesh.vertices, right_vshoe)), np.concatenate((fshoe, fshoe[:, ::-1] + len(right_vshoe) )) )
        #mesh = trimesh.Trimesh(np.concatenate((mesh.vertices, right_vshoe)), np.concatenate((fshoe[:, ::-1], fshoe + len(right_vshoe) )) )
        return mesh

    def pose_mesh(self, mesh, pose, J, v):
        step = self.step
        iters = len(mesh.vertices)//step
        if len(mesh.vertices)%step != 0:
            iters += 1
        for i in range(iters):
            in_verts = torch.FloatTensor(mesh.vertices[i*step:(i+1)*step]).cuda().unsqueeze(0)
            _, out_verts = self.SMPL_Layer.deform_clothed_smpl(pose, J, v, in_verts)
            #out_verts, _ = self.SMPL_Layer.deform_clothed_smpl(pose, J, v, in_verts)
            mesh.vertices[i*step:(i+1)*step] = out_verts.cpu().data.numpy()

        mesh = trimesh.smoothing.filter_laplacian(mesh, lamb=0.5)
        return mesh

    def pose_mesh_lowerbody(self, mesh, pose, beta, J, v):
        step = self.step
        iters = len(mesh.vertices)//step
        if len(mesh.vertices)%step != 0:
            iters += 1
        for i in range(iters):
            in_verts = torch.FloatTensor(mesh.vertices[i*step:(i+1)*step]).cuda()
            out_verts = self.SMPL_Layer.unpose_and_deform_cloth(in_verts, self.Astar_pose, pose, beta, J, v)
            mesh.vertices[i*step:(i+1)*step] = out_verts.cpu().data.numpy()

        mesh = trimesh.smoothing.filter_laplacian(mesh, lamb=0.5)
        return mesh

    def forward(self, model_id, z_cut, z_style, points):
        '''
        Differentiable forward pass of points on any of the models, useful to fit in scans or images.
        Input:
        - model_id: integer of the model to be used: 0=Upperbody, 1=Pants, 2=Skirts, 3=Hair, 4=Shoes
        - Z: Cloth parameters evaluated, tensor of shape [N]
        - points: 3D points in which we will predict distance, with shape [npoints, 3]
        Returns:
        - pred: Signed distance predicted
        '''

        pred = self.models[model_id].forward(z_cut, z_style, points)
        return pred

