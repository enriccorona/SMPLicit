import torch
import json
import sys
import numpy as np
from .util_smpl import batch_global_rigid_transformation, batch_rodrigues, reflect_pose
import torch.nn as nn
import os
import trimesh

class SMPL(nn.Module):
    def __init__(self, model_path, joint_type = 'cocoplus', obj_saveable = False):
        super(SMPL, self).__init__()

        if joint_type not in ['cocoplus', 'lsp']:
            msg = 'unknow joint type: {}, it must be either "cocoplus" or "lsp"'.format(joint_type)
            sys.exit(msg)

        self.model_path = model_path
        self.joint_type = joint_type
        with open(model_path, 'r') as reader:
            model = json.load(reader)
        
        if obj_saveable:
            self.faces = model['f']
        else:
            self.faces = None

        np_v_template = np.array(model['v_template'], dtype = np.float)
        self.register_buffer('v_template', torch.from_numpy(np_v_template).float())
        self.size = [np_v_template.shape[0], 3]

        np_shapedirs = np.array(model['shapedirs'], dtype = np.float)
        self.num_betas = np_shapedirs.shape[-1]
        np_shapedirs = np.reshape(np_shapedirs, [-1, self.num_betas]).T
        self.register_buffer('shapedirs', torch.from_numpy(np_shapedirs).float())

        np_J_regressor = np.array(model['J_regressor'], dtype = np.float)
        self.register_buffer('J_regressor', torch.from_numpy(np_J_regressor).float())

        np_posedirs = np.array(model['posedirs'], dtype = np.float)
        num_pose_basis = np_posedirs.shape[-1]
        np_posedirs = np.reshape(np_posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', torch.from_numpy(np_posedirs).float())

        self.parents = np.array(model['kintree_table'])[0].astype(np.int32)

        np_joint_regressor = np.array(model['cocoplus_regressor'], dtype = np.float)
        if joint_type == 'lsp':
            self.register_buffer('joint_regressor', torch.from_numpy(np_joint_regressor[:, :14]).float())
        else:
            self.register_buffer('joint_regressor', torch.from_numpy(np_joint_regressor).float())

        np_weights = np.array(model['weights'], dtype = np.float)

        vertex_count = np_weights.shape[0] 
        vertex_component = np_weights.shape[1]

        # batch_size = 10
        # np_weights = np.tile(np_weights, (batch_size, 1))
        self.register_buffer('weight', torch.from_numpy(np_weights).float().reshape(-1, vertex_count, vertex_component))

        self.register_buffer('e3', torch.eye(3).float())
        
        self.cur_device = None

    def save_obj(self, verts, obj_mesh_name):
        if not self.faces:
            msg = 'obj not saveable!'
            sys.exit(msg)

        with open(obj_mesh_name, 'w') as fp:
            for v in verts:
                fp.write( 'v {:f} {:f} {:f}\n'.format( v[0], v[1], v[2]) )

            for f in self.faces: # Faces are 1-based, not 0-based in obj files
                fp.write( 'f {:d} {:d} {:d}\n'.format(f[0] + 1, f[1] + 1, f[2] + 1) )

    def forward(self, beta, theta, get_skin = False, theta_in_rodrigues=True):
        device = beta.device
        self.cur_device = torch.device(device.type, device.index)

        num_batch = beta.shape[0]

        v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim = 2)
        if theta_in_rodrigues:
            Rs = batch_rodrigues(theta.view(-1, 3)).view(-1, 24, 3, 3)
        else: #theta is already rotations
            Rs = theta.view(-1,24,3,3)

        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)
        v_posed = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1]) + v_shaped
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, rotate_base = False)

        W=self.weight.expand(num_batch,*self.weight.shape[1:])
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)
        
        v_posed_homo = torch.cat([v_posed, torch.ones(num_batch, v_posed.shape[1], 1, device = self.cur_device)], dim = 2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))

        verts = v_homo[:, :, :3, 0]

        joint_x = torch.matmul(verts[:, :, 0], self.joint_regressor)
        joint_y = torch.matmul(verts[:, :, 1], self.joint_regressor)
        joint_z = torch.matmul(verts[:, :, 2], self.joint_regressor)

        joints = torch.stack([joint_x, joint_y, joint_z], dim = 2)

        if get_skin:
            return verts, joints, Rs
        else:
            return joints

    def deform_clothes_smpl_usingseveralpoints(self, theta, J, v_smpl, v_cloth, neighbors = 3):
        assert len(theta) == 1, 'currently we only support batchsize=1'
        num_batch=1

        device = theta.device
        self.cur_device = torch.device(device.type, device.index)

        Rs = batch_rodrigues(theta.view(-1, 3)).view(-1, 24, 3, 3)
        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)

        pose_params = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1])
        v_posed_smpl = pose_params + v_smpl

        # Calculate closest SMPL vertex for each vertex of the cloth mesh
        with torch.no_grad():
            dists = ((v_smpl.unsqueeze(1) - v_cloth.unsqueeze(2))**2).sum(-1)

            # Calculate closest SMPL N vertices for each vertex of the closest cloth mesh
            correspondance = []
            for i in range(neighbors):
                new_corresponance = torch.argmin(dists, 2)
                dists[0, np.arange(dists.shape[1]), new_correspondance] += 100
                correspondance.append(new_correspondance)
            correspondance = torch.stack(correspondance, -1)

        v_posed_cloth = pose_params[0, correspondance[0]] + v_cloth
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, rotate_base = False)

        W=self.weight.expand(num_batch,*self.weight.shape[1:])
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

        v_posed_homo_smpl = torch.cat([v_posed_smpl, torch.ones(num_batch, v_posed_smpl.shape[1], 1, device = self.cur_device)], dim = 2)
        v_posed_homo_cloth = torch.cat([v_posed_cloth, torch.ones(num_batch, v_posed_cloth.shape[1], 1, device = self.cur_device)], dim = 2)
        v_homo_smpl = torch.matmul(T, torch.unsqueeze(v_posed_homo_smpl, -1))
        applying_T = T[0, correspondance].mean(2)
        # Normalizing average of these T:
        norm = torch.sqrt((applying_T[0, :, :3, :3]**2).sum(-1)).unsqueeze(-1)
        applying_T[0, :, :3, :3] /= norm

        v_homo_cloth = torch.matmul(applying_T, torch.unsqueeze(v_posed_homo_cloth, -1))
        verts_smpl = v_homo_smpl[:, :, :3, 0]
        verts_cloth = v_homo_cloth[:, :, :3, 0]

        return verts_smpl, verts_cloth

    def deform_clothed_smpl_usingseveralpoints(self, theta, J, v_smpl, v_cloth, neighbors = 3):
        assert len(theta) == 1, 'currently we only support batchsize=1'
        num_batch=1

        device = theta.device
        self.cur_device = torch.device(device.type, device.index)

        Rs = batch_rodrigues(theta.view(-1, 3)).view(-1, 24, 3, 3)
        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)

        pose_params = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1])
        v_posed_smpl = pose_params + v_smpl

        # Calculate closest SMPL vertex for each vertex of the cloth mesh
        with torch.no_grad():
            dists = ((v_smpl.unsqueeze(1) - v_cloth.unsqueeze(2))**2).sum(-1)

            # EQUIVALENT TO 'correspondance = torch.argsort(dists, 2)[:, :, :neighbors]' BUT MUCH FASTER:
            correspondance = []
            for i in range(neighbors):
                new_correspondance = torch.argmin(dists, 2)
                dists[0, np.arange(dists.shape[1]), new_correspondance] += 100
                correspondance.append(new_correspondance)
            correspondance = torch.stack(correspondance, -1)

        v_posed_cloth = pose_params[0, correspondance].mean(2) + v_cloth
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, rotate_base = False)

        W=self.weight.expand(num_batch,*self.weight.shape[1:])
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

        v_posed_homo_smpl = torch.cat([v_posed_smpl, torch.ones(num_batch, v_posed_smpl.shape[1], 1, device = self.cur_device)], dim = 2)
        v_posed_homo_cloth = torch.cat([v_posed_cloth, torch.ones(num_batch, v_posed_cloth.shape[1], 1, device = self.cur_device)], dim = 2)
        v_homo_smpl = torch.matmul(T, torch.unsqueeze(v_posed_homo_smpl, -1))
        applying_T = T[0, correspondance].mean(2)
        # Normalizing average of this T:
        norm = torch.sqrt((applying_T[0, :, :3, :3]**2).sum(-1)).unsqueeze(-1)
        applying_T[0, :, :3, :3] /= norm

        v_homo_cloth = torch.matmul(applying_T, torch.unsqueeze(v_posed_homo_cloth, -1))
        verts_smpl = v_homo_smpl[:, :, :3, 0]
        verts_cloth = v_homo_cloth[:, :, :3, 0]

        return verts_smpl, verts_cloth

    def deform_clothed_smpl_usingseveralpoints2(self, theta, J, v_smpl, v_cloth, neighbors = 3):
        assert len(theta) == 1, 'currently we only support batchsize=1'
        num_batch=1

        device = theta.device
        self.cur_device = torch.device(device.type, device.index)

        Rs = batch_rodrigues(theta.view(-1, 3)).view(-1, 24, 3, 3)
        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)

        pose_params = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1])
        v_posed_smpl = pose_params + v_smpl

        # Calculate closest SMPL vertex for each vertex of the cloth mesh
        with torch.no_grad():
            dists = ((v_smpl.unsqueeze(1) - v_cloth.unsqueeze(2))**2).sum(-1)

            # EQUIVALENT TO 'correspondance = torch.argsort(dists, 2)[:, :, :neighbors]' BUT MUCH FASTER:
            correspondance = []
            corresponding_dists = []
            for i in range(neighbors):
                new_correspondance = torch.argmin(dists, 2)
                direct_dists = dists[0, np.arange(dists.shape[1]), new_correspondance[0]]
                dists[0, np.arange(dists.shape[1]), new_correspondance[0]] += 100
                correspondance.append(new_correspondance)
                corresponding_dists.append(direct_dists)
            correspondance = torch.stack(correspondance, -1)
            corresponding_dists = torch.stack(corresponding_dists, -1)

            interpolation = 'linear'
            #interpolation = 'powerfour'
            #interpolation = 'squared'
            if interpolation == 'squared': # NOTE: Consider transforming this from m to mm to avoid numeric issues
                #sum_dists = torch.sqrt(torch.sqrt((corresponding_dists**4).sum(-1).unsqueeze(-1)))
                sum_dists = torch.sqrt((corresponding_dists**2).sum(-1).unsqueeze(-1))
                weights_dists = (sum_dists - corresponding_dists)
                weights_dists = weights_dists/weights_dists.sum(-1).unsqueeze(-1)
                #weights_dists = weights_dists/torch.sqrt((weights_dists**2).sum(-1).unsqueeze(-1))
            elif interpolation == 'powerfour':
                sum_dists = torch.sqrt(torch.sqrt((corresponding_dists**4).sum(-1).unsqueeze(-1)))
                weights_dists = (sum_dists - corresponding_dists)
                weights_dists = weights_dists/weights_dists.sum(-1).unsqueeze(-1)
            elif interpolation == 'linear':
                sum_dists = torch.abs(corresponding_dists).sum(-1).unsqueeze(-1)
                weights_dists = (sum_dists - corresponding_dists)
                weights_dists = weights_dists/weights_dists.sum(-1).unsqueeze(-1)

        v_posed_cloth = (pose_params[0, correspondance]*weights_dists.unsqueeze(0).unsqueeze(-1)).sum(2) + v_cloth
        #v_posed_cloth = pose_params[0, correspondance].mean(2) + v_cloth
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, rotate_base = False)

        W=self.weight.expand(num_batch,*self.weight.shape[1:])
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

        v_posed_homo_smpl = torch.cat([v_posed_smpl, torch.ones(num_batch, v_posed_smpl.shape[1], 1, device = self.cur_device)], dim = 2)
        v_posed_homo_cloth = torch.cat([v_posed_cloth, torch.ones(num_batch, v_posed_cloth.shape[1], 1, device = self.cur_device)], dim = 2)
        v_homo_smpl = torch.matmul(T, torch.unsqueeze(v_posed_homo_smpl, -1))
        applying_T = T[0, correspondance[0, :, 0]]
        #applying_T = (T[0, correspondance]*weights_dists.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)).sum(2)
        #applying_T = T[0, correspondance].mean(2)
        # Normalizing average of this T:
        #norm = torch.sqrt((applying_T[0, :, :3, :3]**2).sum(-1)).unsqueeze(-1)
        #applying_T[0, :, :3, :3] /= norm

        v_homo_cloth = torch.matmul(applying_T, torch.unsqueeze(v_posed_homo_cloth, -1))
        verts_smpl = v_homo_smpl[:, :, :3, 0]
        verts_cloth = v_homo_cloth[:, :, :3, 0]

        return verts_smpl, verts_cloth


    def deform_clothed_smpl(self, theta, J, v_smpl, v_cloth):
        assert len(theta) == 1, 'currently we only support batchsize=1'
        num_batch=1

        device = theta.device
        self.cur_device = torch.device(device.type, device.index)

        Rs = batch_rodrigues(theta.view(-1, 3)).view(-1, 24, 3, 3)
        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)

        pose_params = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1])
        v_posed_smpl = pose_params + v_smpl

        # Calculate closest SMPL vertex for each vertex of the cloth mesh
        with torch.no_grad():
            dists = ((v_smpl.unsqueeze(1) - v_cloth.unsqueeze(2))**2).sum(-1)
            correspondance = torch.argmin(dists, 2)

        v_posed_cloth = pose_params[0, correspondance[0]] + v_cloth
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, rotate_base = False)

        W=self.weight.expand(num_batch,*self.weight.shape[1:])
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

        v_posed_homo_smpl = torch.cat([v_posed_smpl, torch.ones(num_batch, v_posed_smpl.shape[1], 1, device = self.cur_device)], dim = 2)
        v_posed_homo_cloth = torch.cat([v_posed_cloth, torch.ones(num_batch, v_posed_cloth.shape[1], 1, device = self.cur_device)], dim = 2)
        v_homo_smpl = torch.matmul(T, torch.unsqueeze(v_posed_homo_smpl, -1))
        v_homo_cloth = torch.matmul(T[0, correspondance[0]], torch.unsqueeze(v_posed_homo_cloth, -1))
        verts_smpl = v_homo_smpl[:, :, :3, 0]
        verts_cloth = v_homo_cloth[:, :, :3, 0]

        return verts_smpl, verts_cloth
   
    def deform_clothed_smpl_w_normals(self, theta, J, v_smpl, v_cloth, v_normals):
        assert len(theta) == 1, 'currently we only support batchsize=1'
        num_batch=1

        device = theta.device
        self.cur_device = torch.device(device.type, device.index)

        Rs = batch_rodrigues(theta.view(-1, 3)).view(-1, 24, 3, 3)
        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)

        pose_params = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1])
        v_posed_smpl = pose_params + v_smpl

        # Calculate closest SMPL vertex for each vertex of the cloth mesh
        with torch.no_grad():
            dists = ((v_smpl.unsqueeze(1) - v_cloth.unsqueeze(2))**2).sum(-1)
            correspondance = torch.argmin(dists, 2)

        v_posed_cloth = pose_params[0, correspondance[0]] + v_cloth
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, rotate_base = False)

        W=self.weight.expand(num_batch,*self.weight.shape[1:])
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

        v_posed_homo_smpl = torch.cat([v_posed_smpl, torch.ones(num_batch, v_posed_smpl.shape[1], 1, device = self.cur_device)], dim = 2)
        v_posed_homo_cloth = torch.cat([v_posed_cloth, torch.ones(num_batch, v_posed_cloth.shape[1], 1, device = self.cur_device)], dim = 2)
        v_homo_smpl = torch.matmul(T, torch.unsqueeze(v_posed_homo_smpl, -1))
        v_homo_cloth = torch.matmul(T[0, correspondance[0]], torch.unsqueeze(v_posed_homo_cloth, -1))

        v_normals_posed = torch.cat([v_normals, torch.ones(num_batch, v_normals.shape[1], 1, device=self.cur_device)], dim=2)
        v_normals_posed = torch.matmul(T[0, correspondance[0]], torch.unsqueeze(v_normals_posed, -1))

        verts_smpl = v_homo_smpl[:, :, :3, 0]
        verts_cloth = v_homo_cloth[:, :, :3, 0]
        verts_normals = v_normals_posed[:, :, :3, 0]

        return verts_smpl, verts_cloth, verts_normals
   
    def deform_clothed_smpl_consistent(self, theta, J, v_smpl, v_cloth, normals_cloth, thresh=0.2):
        assert len(theta) == 1, 'currently we only support batchsize=1'
        num_batch=1

        device = theta.device
        self.cur_device = torch.device(device.type, device.index)

        Rs = batch_rodrigues(theta.view(-1, 3)).view(-1, 24, 3, 3)
        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)

        pose_params = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1])
        v_posed_smpl = pose_params + v_smpl

        # Calculate closest SMPL vertex for each vertex of the cloth mesh
        with torch.no_grad():
            dists = ((v_smpl.unsqueeze(1) - v_cloth.unsqueeze(2))**2).sum(-1)
            normals_smpl = trimesh.Trimesh(v_smpl.cpu().data.numpy()[0], self.faces, process=False).vertex_normals
            angle_bw_points = np.dot(normals_cloth, normals_smpl.T)
            #angles = np.arccos(angle_bw_points)

            # Basically removing those that are not appropriate:
            dists = dists[0]
            dists[angle_bw_points < thresh] += 1000
            #dists[np.abs(angle_bw_points) < 0.7] += 1000

            correspondance = torch.argmin(dists, 1).unsqueeze(0)

        v_posed_cloth = pose_params[0, correspondance[0]] + v_cloth
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, rotate_base = False)

        W=self.weight.expand(num_batch,*self.weight.shape[1:])
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

        v_posed_homo_smpl = torch.cat([v_posed_smpl, torch.ones(num_batch, v_posed_smpl.shape[1], 1, device = self.cur_device)], dim = 2)
        v_posed_homo_cloth = torch.cat([v_posed_cloth, torch.ones(num_batch, v_posed_cloth.shape[1], 1, device = self.cur_device)], dim = 2)
        v_homo_smpl = torch.matmul(T, torch.unsqueeze(v_posed_homo_smpl, -1))
        v_homo_cloth = torch.matmul(T[0, correspondance[0]], torch.unsqueeze(v_posed_homo_cloth, -1))
        verts_smpl = v_homo_smpl[:, :, :3, 0]
        verts_cloth = v_homo_cloth[:, :, :3, 0]

        return verts_smpl, verts_cloth

    def normalization_cloth_beta(self, v, beta, v_smpl=None):
        if type(v_smpl) == type(None):
            # JUST GETTING SMPL MODEL ON T-Pose:
            v_smpl = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
        else:
            v_smpl = v_smpl.unsqueeze(0)

        with torch.no_grad():
            dists = ((v_smpl - v.unsqueeze(1))**2).sum(-1)
            correspondance = torch.argmin(dists, 1)
            # NOTE: Matmul is very highdimensional. Reduce only getting those that are necessary:
            # NOTE: For dense meshes is actually faster to run the matmul first and get correspnding rows later
            beta_normalization = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1])[0][correspondance]
            #v_normalized = v - beta_normalization
        return beta_normalization.cpu().data.numpy()

    def expand_cloth_beta(self, v, beta, new_beta, v_smpl=None):
        if type(v_smpl) == type(None):
            # JUST GETTING SMPL MODEL ON T-Pose:
            v_smpl = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
        else:
            v_smpl = v_smpl.unsqueeze(0)

        with torch.no_grad():
            dists = ((v_smpl - v.unsqueeze(1))**2).sum(-1)
            correspondance = torch.argmin(dists, 1)
            # NOTE: Matmul is very highdimensional. Reduce only getting those that are necessary:
            # NOTE: For dense meshes it's actually faster to run the matmul first and get correspnding rows later
            beta_normalization = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1])[0][correspondance]
            beta_addition = torch.matmul(new_beta, self.shapedirs).view(-1, self.size[0], self.size[1])[0][correspondance]
            v_normalized = v - beta_normalization + beta_addition
        return v_normalized

    def unpose_and_deform_cloth(self, v_cloth_posed, theta_from, theta_to, beta, Jsmpl, vsmpl, theta_in_rodrigues=True):
        ### UNPOSE:
        device = theta_from.device
        self.cur_device = torch.device(device.type, device.index)
        num_batch = beta.shape[0]

        v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim = 2)
        if theta_in_rodrigues:
            Rs = batch_rodrigues(theta_from.view(-1, 3)).view(-1, 24, 3, 3)
        else: #theta is already rotations
            Rs = theta_from.view(-1,24,3,3)

        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)
        pose_displ = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1])
        v_posed = pose_displ + v_shaped
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, rotate_base = False)

        W=self.weight.expand(num_batch,*self.weight.shape[1:])
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

        v_posed_homo = torch.cat([v_posed, torch.ones(num_batch, v_posed.shape[1], 1, device = self.cur_device)], dim = 2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))

        v_smpl = v_homo[:, :, :3, 0]
        with torch.no_grad():
            dists = ((v_smpl - v_cloth_posed.unsqueeze(1))**2).sum(-1)
            correspondance = torch.argmin(dists, 1)

        invT = torch.inverse(T[0, correspondance])
        v = torch.cat([v_cloth_posed, torch.ones(len(v_cloth_posed), 1, device=self.cur_device)], 1)
        v = torch.matmul(invT, v.unsqueeze(-1))[:, :3, 0]
        unposed_v = v - pose_displ[0, correspondance]
        
        ### REPOSE:
        Rs = batch_rodrigues(theta_to.view(-1, 3)).view(-1, 24, 3, 3)
        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)

        pose_params = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1])
        v_posed_cloth = pose_params[:, correspondance] + unposed_v.unsqueeze(0)
        self.J_transformed, A = batch_global_rigid_transformation(Rs, Jsmpl, self.parents, rotate_base = False)

        W=self.weight.expand(num_batch,*self.weight.shape[1:])
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

        v_posed_homo_cloth = torch.cat([v_posed_cloth, torch.ones(num_batch, v_posed_cloth.shape[1], 1, device = self.cur_device)], dim = 2)
        v_homo_cloth = torch.matmul(T[0, correspondance], torch.unsqueeze(v_posed_homo_cloth, -1))
        verts_cloth = v_homo_cloth[:, :, :3, 0]
        return verts_cloth[0]

    def unpose_and_deform_cloth_w_normals(self, v_cloth_posed, v_normals, theta_from, theta_to, beta, Jsmpl, vsmpl):
        device = theta_from.device
        self.cur_device = torch.device(device.type, device.index)
        num_batch = beta.shape[0]

        v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim = 2)
        Rs = batch_rodrigues(theta_from.view(-1, 3)).view(-1, 24, 3, 3)

        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)
        pose_displ = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1])
        v_posed = pose_displ + v_shaped
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, rotate_base = False)

        W=self.weight.expand(num_batch,*self.weight.shape[1:])
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

        v_posed_homo = torch.cat([v_posed, torch.ones(num_batch, v_posed.shape[1], 1, device = self.cur_device)], dim = 2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))

        v_smpl = v_homo[:, :, :3, 0]
        with torch.no_grad():
            dists = ((v_smpl - v_cloth_posed.unsqueeze(1))**2).sum(-1)
            correspondance = torch.argmin(dists, 1)

        invT = torch.inverse(T[0, correspondance])
        v = torch.cat([v_cloth_posed, torch.ones(len(v_cloth_posed), 1, device=self.cur_device)], 1)
        v = torch.matmul(invT, v.unsqueeze(-1))[:, :3, 0]
        unposed_v = v - pose_displ[0, correspondance]
        # Not applying T to normals since model was trained on normals from T-Pose
        
        ### REPOSE:
        Rs = batch_rodrigues(theta_to.view(-1, 3)).view(-1, 24, 3, 3)
        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)

        pose_params = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1])
        v_posed_cloth = pose_params[:, correspondance] + unposed_v.unsqueeze(0)
        self.J_transformed, A = batch_global_rigid_transformation(Rs, Jsmpl, self.parents, rotate_base = False)

        W=self.weight.expand(num_batch,*self.weight.shape[1:])
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

        v_posed_homo_cloth = torch.cat([v_posed_cloth, torch.ones(num_batch, v_posed_cloth.shape[1], 1, device = self.cur_device)], dim = 2)
        v_homo_cloth = torch.matmul(T[0, correspondance], torch.unsqueeze(v_posed_homo_cloth, -1))
        verts_cloth = v_homo_cloth[:, :, :3, 0]

        v_normals_posed = torch.cat([v_normals.unsqueeze(0), torch.ones(num_batch, v_normals.shape[0], 1, device=self.cur_device)], dim=2)
        v_normals_posed = torch.matmul(T[0, correspondance], torch.unsqueeze(v_normals_posed, -1))
        v_normals_posed = v_normals_posed[:, :, :3, 0]

        return verts_cloth[0], v_normals_posed[0]


    def unpose_and_deform_cloth_w_normals2(self, v_cloth_posed, v_normals, theta_from, theta_to, beta, Jsmpl, vsmpl, v_normals_smooth, theta_in_rodrigues=True):
        device = theta_from.device
        self.cur_device = torch.device(device.type, device.index)
        num_batch = beta.shape[0]

        v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim = 2)
        if theta_in_rodrigues:
            Rs = batch_rodrigues(theta_from.view(-1, 3)).view(-1, 24, 3, 3)
        else: #theta is already rotations
            Rs = theta_from.view(-1,24,3,3)

        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)
        pose_displ = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1])
        v_posed = pose_displ + v_shaped
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, rotate_base = False)

        W=self.weight.expand(num_batch,*self.weight.shape[1:])
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

        v_posed_homo = torch.cat([v_posed, torch.ones(num_batch, v_posed.shape[1], 1, device = self.cur_device)], dim = 2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))

        v_smpl = v_homo[:, :, :3, 0]
        angle_thresh = 0.1
        with torch.no_grad():
            dists = ((v_smpl - v_cloth_posed.unsqueeze(1))**2).sum(-1)
            normals_smpl = trimesh.Trimesh(v_smpl.cpu().data.numpy()[0], self.faces, process=False).vertex_normals
            angle_bw_points = np.dot(v_normals_smooth, normals_smpl.T)
            dists[angle_bw_points < angle_thresh] += 1000
            correspondance = torch.argmin(dists, 1)

        invT = torch.inverse(T[0, correspondance])
        v = torch.cat([v_cloth_posed, torch.ones(len(v_cloth_posed), 1, device=self.cur_device)], 1)
        v = torch.matmul(invT, v.unsqueeze(-1))[:, :3, 0]
        unposed_v = v - pose_displ[0, correspondance]
        # Not applying T to normals since model was trained on normals from T-Pose
        
        ### REPOSE:
        Rs = batch_rodrigues(theta_to.view(-1, 3)).view(-1, 24, 3, 3)
        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)

        pose_params = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1])
        v_posed_cloth = pose_params[:, correspondance] + unposed_v.unsqueeze(0)
        self.J_transformed, A = batch_global_rigid_transformation(Rs, Jsmpl, self.parents, rotate_base = False)

        W=self.weight.expand(num_batch,*self.weight.shape[1:])
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

        v_posed_homo_cloth = torch.cat([v_posed_cloth, torch.ones(num_batch, v_posed_cloth.shape[1], 1, device = self.cur_device)], dim = 2)
        v_homo_cloth = torch.matmul(T[0, correspondance], torch.unsqueeze(v_posed_homo_cloth, -1))
        verts_cloth = v_homo_cloth[:, :, :3, 0]

        v_normals_posed = torch.cat([v_normals.unsqueeze(0), torch.ones(num_batch, v_normals.shape[0], 1, device=self.cur_device)], dim=2)
        v_normals_posed = torch.matmul(T[0, correspondance], torch.unsqueeze(v_normals_posed, -1))
        v_normals_posed = v_normals_posed[:, :, :3, 0]

        return verts_cloth[0], v_normals_posed[0]


    def unnormalize_cloth_pose(self, v_cloth_posed, theta, beta, theta_in_rodrigues=True):
        device = theta.device
        self.cur_device = torch.device(device.type, device.index)
        num_batch = beta.shape[0]

        v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim = 2)
        if theta_in_rodrigues:
            Rs = batch_rodrigues(theta.view(-1, 3)).view(-1, 24, 3, 3)
        else: #theta is already rotations
            Rs = theta.view(-1,24,3,3)

        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)
        pose_displ = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1])
        v_posed = pose_displ + v_shaped
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, rotate_base = False)

        W=self.weight.expand(num_batch,*self.weight.shape[1:])
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)
        
        v_posed_homo = torch.cat([v_posed, torch.ones(num_batch, v_posed.shape[1], 1, device = self.cur_device)], dim = 2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))

        v_smpl = v_homo[:, :, :3, 0]
        with torch.no_grad():
            dists = ((v_smpl - v_cloth_posed.unsqueeze(1))**2).sum(-1)
            correspondance = torch.argmin(dists, 1)

        invT = torch.inverse(T[0, correspondance])
        v = torch.cat([v_cloth_posed, torch.ones(len(v_cloth_posed), 1, device=self.cur_device)], 1)
        v = torch.matmul(invT, v.unsqueeze(-1))[:, :3, 0]
        unposed_v = v - pose_displ[0, correspondance]

        return unposed_v

#    def unnormalize_cloth_beta(self, v, beta, v_smpl=None):
#        if type(v_smpl) == type(None):
#            # JUST GETTING SMPL MODEL ON T-Pose:
#            v_smpl = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
#        else:
#            v_smpl = v_smpl.unsqueeze(0)
#
#        with torch.no_grad():
#            dists = ((v_smpl - v.unsqueeze(1))**2).sum(-1)
#            correspondance = torch.argmin(dists, 1)
#            # NOTE: Matmul is very highdimensional. Reduce only getting those that are necessary:
#            # NOTE: For dense meshes is actually faster to run the matmul first and get correspnding rows later
#            beta_normalization = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1])[0][correspondance]
#            v_normalized = v + beta_normalization
#        return v_normalized

    def skeleton(self,beta,require_body=False):
        num_batch = beta.shape[0]
        v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim = 2)
        if require_body:
            return J, v_shaped
        else:
            return J

def getSMPL():
    return SMPL(os.path.normpath(os.path.join(os.path.dirname(__file__),'model/neutral_smpl_with_cocoplus_reg.txt')), obj_saveable = True)
def getTmpFile():
    return os.path.join(os.path.dirname(__file__),'hello_smpl.obj')

if __name__ == '__main__':
    device = torch.device('cuda', 1)
    # smpl = SMPL('/home/jby/pytorch_ext/smpl_pytorch/model/neutral_smpl_with_cocoplus_reg.txt',obj_saveable=True)
    smpl = SMPL(os.path.join(os.path.dirname(__file__),'model/neutral_smpl_with_cocoplus_reg.txt'), obj_saveable = True).to(device)
    pose= np.array([
            1.22162998e+00,   1.17162502e+00,   1.16706634e+00,
            -1.20581151e-03,   8.60930011e-02,   4.45963144e-02,
            -1.52801601e-02,  -1.16911056e-02,  -6.02894090e-03,
            1.62427306e-01,   4.26302850e-02,  -1.55304456e-02,
            2.58729942e-02,  -2.15941742e-01,  -6.59851432e-02,
            7.79098943e-02,   1.96353287e-01,   6.44420758e-02,
            -5.43042570e-02,  -3.45508829e-02,   1.13200583e-02,
            -5.60734887e-04,   3.21716577e-01,  -2.18840033e-01,
            -7.61821344e-02,  -3.64610642e-01,   2.97633410e-01,
            9.65453908e-02,  -5.54007106e-03,   2.83410680e-02,
            -9.57194716e-02,   9.02515948e-02,   3.31488043e-01,
            -1.18847653e-01,   2.96623230e-01,  -4.76809204e-01,
            -1.53382001e-02,   1.72342166e-01,  -1.44332021e-01,
            -8.10869411e-02,   4.68325168e-02,   1.42248288e-01,
            -4.60898802e-02,  -4.05981280e-02,   5.28727695e-02,
            3.20133418e-02,  -5.23784310e-02,   2.41559884e-03,
            -3.08033824e-01,   2.31431410e-01,   1.62540793e-01,
            6.28208935e-01,  -1.94355965e-01,   7.23800480e-01,
            -6.49612308e-01,  -4.07179184e-02,  -1.46422181e-02,
            4.51475441e-01,   1.59122205e+00,   2.70355493e-01,
            2.04248756e-01,  -6.33800551e-02,  -5.50178960e-02,
            -1.00920045e+00,   2.39532292e-01,   3.62904727e-01,
            -3.38783532e-01,   9.40650925e-02,  -8.44506770e-02,
            3.55101633e-03,  -2.68924050e-02,   4.93676625e-02],dtype = np.float)
        
    beta = np.array([-0.25349993,  0.25009069,  0.21440795,  0.78280628,  0.08625954,
            0.28128183,  0.06626327, -0.26495767,  0.09009246,  0.06537955 ])

    vbeta = torch.tensor(np.array([beta])).float().to(device)
    vpose = torch.tensor(np.array([pose])).float().to(device)

    verts, j, r = smpl(vbeta, vpose, get_skin = True)

    smpl.save_obj(verts[0].cpu().numpy(), './mesh.obj')

    rpose = reflect_pose(pose)
    vpose = torch.tensor(np.array([rpose])).float().to(device)
    
    verts, j, r = smpl(vbeta, vpose, get_skin = True)
    smpl.save_obj(verts[0].cpu().numpy(), './rmesh.obj')

