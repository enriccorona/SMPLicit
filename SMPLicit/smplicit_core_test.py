import torch
import numpy as np
from .utils.sdf import create_grid, eval_grid_octree, eval_grid
from skimage import measure
from .network import Network

import trimesh
import socket


class Model():
    def __init__(self, filename, n_z_cut, n_z_style, num_clusters, name_clusters, b_min, b_max, resolution, thresh=-0.05):
        self.filename = filename
        self.n_z_cut = n_z_cut
        self.n_z_style = n_z_style
        self.num_clusters = num_clusters
        self.clusters = np.load(name_clusters, allow_pickle=True)
        self.b_min = b_min
        self.b_max = b_max
        self.resolution = resolution
        self.thresh = thresh

        # create networks
        self._init_create_networks()

        self.load()


    def _init_create_networks(self):
        # generator network
        self._G = self._create_generator()
        if torch.cuda.is_available():
            self._G.cuda()

    def _create_generator(self):
        return Network(n_z_style=self.n_z_style, point_pos_size=self.num_clusters*3, output_dim=1, n_z_cut=self.n_z_cut)

    def set_eval(self):
        self._G.eval()

    def reconstruct(self, z, smpl_trianglemesh):
        resolution = self.resolution
        print(resolution)
        thresh = self.thresh
        b_min = np.array(self.b_min)
        b_max = np.array(self.b_max)

        z_cut = z[:self.n_z_cut]
        z_style = z[self.n_z_cut:]

        coords, mat = create_grid(resolution, resolution, resolution, b_min, b_max)

        if not torch.is_tensor(z_cut):
            z_cut = torch.FloatTensor(z_cut).cuda()
        if not torch.is_tensor(z_style):
            z_style = torch.FloatTensor(z_style).cuda()

        z_cut = z_cut.cuda().reshape(1, -1)
        z_style = z_style.cuda().reshape(1, -1)

        smpl_points = smpl_trianglemesh.vertices
        smpl_points = smpl_points[self.clusters[self.num_clusters]]

        def eval_func(points):
            points = torch.FloatTensor(points).transpose(1, 0).cuda()
            dist = points[:, np.newaxis] - smpl_points[np.newaxis]
            dist = dist.reshape(1, len(points), -1)

            if len(z_style) == 0:
                pred = self._G(z_cut, dist)*-100
            else:
                pred = self._G(z_cut, z_style, dist)*-100
            #pred = self._G(z_cut, z_style, dist)*-100
            return pred.cpu().data.numpy()

        sdf = eval_grid_octree(coords, eval_func, threshold=0.01, num_samples=10000, init_resolution=16)

        verts, faces, normals, values = measure.marching_cubes_lewiner(sdf, thresh)
        cloth_mesh = trimesh.Trimesh(np.float64(verts), faces[:, ::-1])
        cloth_mesh.vertices /= resolution
        cloth_mesh.vertices *= (b_max - b_min)[None]
        cloth_mesh.vertices += b_min[None]

        smooth = True
        if smooth:
            smooth_mesh = trimesh.smoothing.filter_laplacian(cloth_mesh, lamb=0.5)
            if not np.isnan(smooth_mesh.vertices).any():
                cloth_mesh = smooth_mesh

        self.trim_mesh = cloth_mesh
        return self.trim_mesh

    def load(self):
        # load G
        self._G.load_state_dict(torch.load(self.filename))
