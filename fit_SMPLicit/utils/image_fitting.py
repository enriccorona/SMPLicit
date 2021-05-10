import numpy as np
import trimesh
import torch
import cv2
from .sdf import create_grid, eval_grid_octree, eval_grid
from . import projection as projection_utils
import tqdm
from torch import nn

# Define optimization class
# TODO: Add initialization option and way to retrieve initialization to check the difference
class OptimizationCloth(nn.Module):
    def __init__(self, n_style_params=12, n_shape_params=12, initialize_style=None):
        super(OptimizationCloth, self).__init__()

        if type(initialize_style) is type(None):
            self.style = nn.Parameter((torch.rand(1, n_style_params).cuda() - 0.5)/10)
        else:
            self.style = nn.Parameter(torch.FloatTensor(initialize_style).unsqueeze(0).cuda())
        self.shape = nn.Parameter((torch.rand(1, n_shape_params).cuda() - 0.5)/10)

    def forward(self):
        return self.style, self.shape

class OptimizationSMPL(nn.Module):
    def __init__(self, pose, beta, trans):
        super(OptimizationSMPL, self).__init__()
        self.pose_factor = 1.0
        self.beta_factor = 0.3
        self.trans_factor = 0.1

        self.pose = nn.Parameter(torch.FloatTensor(pose).cuda()/self.pose_factor)
        self.beta = nn.Parameter(torch.FloatTensor(beta).cuda()/self.beta_factor)
        self.trans = nn.Parameter(torch.FloatTensor(trans).cuda()/self.trans_factor)

    def forward(self):
        return self.pose*self.pose_factor, self.beta*self.beta_factor, self.trans*self.trans_factor

def render_image_projection_multiperson_wrenderer(input_image, posed_meshes, posed_normals, colors, camScale, camTrans, topleft, scale_ratio, mode='rgb', view='cam', renderer=None):
    #renderer = meshRenderer()
    renderer.setRenderMode('geo')
    renderer.offscreenMode(True)

    renderer.setWindowSize(input_image.shape[1], input_image.shape[0])
    renderer.setBackgroundTexture(input_image)
    renderer.setViewportSize(input_image.shape[1], input_image.shape[0])

    # self.renderer.add_mesh(meshList[0]['ver'],meshList[0]['f'])
    renderer.clear_mesh()
    for mesh_index in range(len(colors)): #TODO: DO THIS FOR EVERYONE IN THE IMAGE:
        vertices_2d = project_points(posed_meshes[mesh_index].vertices, camScale[mesh_index], camTrans[mesh_index], topleft[mesh_index], scale_ratio[mesh_index], input_image)
        vertices_2d[:,0] -= input_image.shape[1]*0.5
        vertices_2d[:,1] -= input_image.shape[0]*0.5
        if mode == 'normals':
            color = posed_normals[mesh_index]*0.5 + 0.5
            fake_normals = np.zeros_like(posed_normals[mesh_index])
            if view == 'cam':
                fake_normals[:, 2] = -1
            elif view == 'side':
                fake_normals[:, 0] = -1

            renderer.add_mesh(vertices_2d, posed_meshes[mesh_index].faces, fake_normals, color=color)
        elif mode == 'rgb':
            if len(colors[mesh_index]) > 10:
                renderer.add_mesh(vertices_2d, posed_meshes[mesh_index].faces, posed_normals[mesh_index], np.array(colors[mesh_index])[:, :3]/255.0)
            else:
                renderer.add_mesh(vertices_2d, posed_meshes[mesh_index].faces, posed_normals[mesh_index], np.array(colors[mesh_index])[:3]/255.0)
        elif mode == 'depth':
            fake_normals = np.zeros_like(posed_meshes[mesh_index].vertices)
            if view == 'cam':
                fake_normals[:, 2] = -1
            elif view == 'side':
                fake_normals[:, 0] = -1
            color = colors[mesh_index]
            renderer.add_mesh(vertices_2d, posed_meshes[mesh_index].faces, fake_normals, color=color)
        else:
            print(mode)
            raise('unknown projection mode')

    if view=='cam':
        renderer.showBackground(True)
    else:
        renderer.showBackground(False)
    renderer.setWorldCenterBySceneCenter()
    renderer.setCameraViewMode(view)

    renderer.display()
    renderImg = renderer.get_screen_color_ibgr()
    return renderImg

def save_video(images, path, freeze_first=50, freeze_last=50, framerate = 5):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(path+'.mp4',fourcc, framerate, (images[0].shape[1], images[0].shape[0]))

    # Leave first frame for some time
    for i in range(freeze_first):
        #frame = cv2.flip(images[0],0)
        frame = images[0]
        out.write(frame)

    # Pass through video:
    for i in range(len(images)):
        #frame = cv2.flip(images[i],0)
        frame = images[i]
        out.write(frame)

    # Leave last frame for around a second
    for i in range(freeze_last):
        out.write(frame)

    out.release()

def project_points(vertices_3d, camScale, camTrans, topleft, scale_ratio, input_image):
    # 1. SMPL -> 2D bbox
    vertices_2d = projection_utils.convert_smpl_to_bbox(vertices_3d, camScale, camTrans)

    # 2. 2D bbox -> original 2D image
    vertices_2d = projection_utils.convert_bbox_to_oriIm(
        vertices_2d, scale_ratio, topleft, input_image.shape[1], input_image.shape[0])

    return vertices_2d

def remove_outside_vertices(verts_2d, input_image):
    valid = verts_2d[:, 0] > 0
    valid &= verts_2d[:, 1] > 0
    valid &= verts_2d[:, 0] < input_image.shape[1]
    valid &= verts_2d[:, 1] < input_image.shape[0]

    verts_2d = verts_2d[valid]
    return verts_2d, valid

def render_depth_image(mesh_smpl, camScale, camTrans, topleft, scale_ratio, input_image):
    fake_colors = mesh_smpl.vertices[:, 2:3].repeat(3, 1)
    min_depth = fake_colors.min()
    max_depth = fake_colors.max()
    fake_colors = (fake_colors - min_depth)/(max_depth-min_depth)
    depth_image_smpl = render_image_projection(input_image*0, [mesh_smpl], [], [fake_colors], camScale, camTrans, topleft, scale_ratio, mode='depth')[:, :, 0]/255.0
    mask = depth_image_smpl == 0
    depth_image_smpl = depth_image_smpl*(max_depth - min_depth) + min_depth
    depth_image_smpl[mask] = 0
    return depth_image_smpl

def unpose_and_deform_cloth(model_trimesh, pose_from, pose_to, beta, J, v, SMPL_Layer, step=10000):
    SMPL_Layer = SMPL_Layer.cpu()
    iters = len(model_trimesh.vertices)//step
    if len(model_trimesh.vertices)%step != 0:
        iters += 1
    for i in range(iters):
        in_verts = torch.FloatTensor(model_trimesh.vertices[i*step:(i+1)*step])
        verts = SMPL_Layer.unpose_and_deform_cloth(in_verts, pose_from, pose_to, beta.cpu(), J.cpu(), v.cpu())
        model_trimesh.vertices[step*i:step*(i+1)] = verts.cpu().data.numpy()
    SMPL_Layer = SMPL_Layer.cuda()
    return model_trimesh

def unpose_and_deform_cloth_tensor(vertices_tensor, pose_from, pose_to, beta, J, v, SMPL_Layer, step=1000):
    #SMPL_Layer = SMPL_Layer.cpu()
    iters = len(vertices_tensor)//step
    if len(vertices_tensor)%step != 0:
        iters += 1
    posed_verts = []
    for i in range(iters):
        in_verts = vertices_tensor[i*step:(i+1)*step]#.cpu()
        verts = SMPL_Layer.unpose_and_deform_cloth(in_verts, pose_from, pose_to, beta, J, v)
        posed_verts.append(verts)
    return torch.cat(posed_verts)

def unpose_and_deform_cloth_w_normals(model_trimesh, pose_from, pose_to, beta, J, v, SMPL_Layer, normals, step=10000):
    SMPL_Layer = SMPL_Layer.cpu()
    for i in range(len(model_trimesh.vertices)//step + 1):
        in_verts = torch.FloatTensor(model_trimesh.vertices[i*step:(i+1)*step])
        in_normals = torch.FloatTensor(normals[i*step:(i+1)*step])
        verts, out_normals = SMPL_Layer.unpose_and_deform_cloth_w_normals(in_verts, in_normals, pose_from, pose_to, beta.cpu(), J.cpu(), v.cpu())
        model_trimesh.vertices[step*i:step*(i+1)] = verts.cpu().data.numpy()
        normals[step*i:step*(i+1)] = out_normals.cpu().data.numpy()

    normals = normals/np.linalg.norm(normals, axis=1)[:, None]
    SMPL_Layer = SMPL_Layer.cuda()
    return model_trimesh, normals

def batch_posing(model_trimesh, pose, J, v, SMPL_Layer, step=10000):
    SMPL_Layer = SMPL_Layer.cpu()
    for i in range(len(model_trimesh.vertices)//step + 1):
        vertices_smpl, deformed_v = SMPL_Layer.deform_clothed_smpl(pose, J.cpu(), v.cpu(), torch.FloatTensor(model_trimesh.vertices[step*i:step*(i+1)]).unsqueeze(0))
        # TODO: This should be more consistent?
        #vertices_smpl, deformed_v = SMPL_Layer.deform_clothed_smpl_usingseveralpoints(pose, J, v, torch.FloatTensor(model_trimesh.vertices[step*i:step*(i+1)]).unsqueeze(0), neighbors=3)
        model_trimesh.vertices[step*i:step*(i+1)] = deformed_v.cpu().data.numpy()[0]
    SMPL_Layer = SMPL_Layer.cuda()
    return model_trimesh

def batch_posing_w_normals(model_trimesh, normals, pose, J, v, SMPL_Layer, step=10000):
    SMPL_Layer = SMPL_Layer.cpu()
    for i in range(len(model_trimesh.vertices)//step + 1):
        in_verts = torch.FloatTensor(model_trimesh.vertices[step*i:step*(i+1)]).unsqueeze(0)
        in_norms = torch.FloatTensor(normals[step*i:step*(i+1)]).unsqueeze(0)
        vertices_smpl, deformed_v, out_normals= SMPL_Layer.deform_clothed_smpl_w_normals(pose, J.cpu(), v.cpu(), in_verts, in_norms)
        # Try consistent too:
        #vertices_smpl, deformed_v = SMPL_Layer.deform_clothed_smpl_usingseveralpoints(pose, J, v, torch.FloatTensor(model_trimesh.vertices[step*i:step*(i+1)]).unsqueeze(0), neighbors=3)
        model_trimesh.vertices[step*i:step*(i+1)] = deformed_v.cpu().data.numpy()[0]
        normals[step*i:step*(i+1)] = out_normals.cpu().data.numpy()
    SMPL_Layer = SMPL_Layer.cuda()
    normals = normals/np.linalg.norm(normals, axis=1)[:, None]

    return model_trimesh, normals

# RENDERER CLASS:
from OpenGL.GLUT import *
from OpenGL.GLU import *
from .shaders.framework import *
from .glRenderer import glRenderer
_glut_window = None

class meshRenderer(glRenderer):

    def __init__(self, width=1600, height=1200, name='GL Renderer',
                render_mode ="normal",  #color, geo, normal
                color_size=1, ms_rate=1):

        self.render_mode = render_mode
        self.program_files ={}
        self.program_files['color'] = ['utils/shaders/simple140.fs', 'utils/shaders/simple140.vs']
        self.program_files['normal'] = ['utils/shaders/normal140.fs', 'utils/shaders/normal140.vs']
        self.program_files['geo'] = ['utils/shaders/colorgeo140.fs', 'utils/shaders/colorgeo140.vs']

        glRenderer.__init__(self, width, height, name, self.program_files[render_mode], color_size, ms_rate)

    def setRenderMode(self, render_mode):
        """
        Set render mode among ['color', 'normal', 'geo']
        """
        if self.render_mode == render_mode:
            return

        self.render_mode = render_mode
        self.initShaderProgram(self.program_files[render_mode])


    def drawMesh(self):
        if self.vertex_dim is None:
            return
        # self.draw_init()

        glColor3f(1,1,0)
        glUseProgram(self.program)

        mvMat = glGetFloatv(GL_MODELVIEW_MATRIX)
        pMat = glGetFloatv(GL_PROJECTION_MATRIX)
        # mvpMat = pMat*mvMat

        self.model_view_matrix = mvMat
        self.projection_matrix = pMat

        glUniformMatrix4fv(self.model_mat_unif, 1, GL_FALSE, self.model_view_matrix)
        glUniformMatrix4fv(self.persp_mat_unif, 1, GL_FALSE, self.projection_matrix)

        # Handle vertex buffer
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, self.vertex_dim, GL_DOUBLE, GL_FALSE, 0, None)

        # # Handle normal buffer
        glBindBuffer(GL_ARRAY_BUFFER, self.normal_buffer)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_DOUBLE, GL_FALSE, 0, None)

        # # Handle color buffer
        glBindBuffer(GL_ARRAY_BUFFER, self.color_buffer)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_DOUBLE, GL_FALSE, 0, None)

        if True:#self.meshindex_data:
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.index_buffer)           #Note "GL_ELEMENT_ARRAY_BUFFER" instead of GL_ARRAY_BUFFER
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.meshindex_data, GL_STATIC_DRAW)

        # glDrawArrays(GL_TRIANGLES, 0, self.n_vertices)
        glDrawElements(GL_TRIANGLES, len(self.meshindex_data), GL_UNSIGNED_INT, None)       #For index array (mesh face data)
        glDisableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glUseProgram(0)
