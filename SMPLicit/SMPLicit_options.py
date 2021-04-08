import argparse
import torch
import os
import numpy as np

# HUMAN PARSING LABELS:
# 1 -> Hat
# 2 -> Hair
# 3 -> Glove
# 4 -> Sunglasses,
# 5 -> Upper-Clothes,
# 6 -> Dress,
# 7 -> Coat,
# 8 -> Socks,
# 9 -> Pants,
# 10 -> Torso-Skin
# 11 -> Scarf
# 12 -> Skirt
# 13 -> Face
# 14 -> Left Arm
# 15 -> Right Arm
# 16 -> Left Leg
# 17 -> Right Leg
# 18 -> Left Shoe
# 19 -> Right Shoe

class Options():
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self._initialized = False

    def initialize(self):
        # Upper body options:
        self._parser.add_argument('--upperbody_model', type=str, default='bcnet_sdf_dist2eachcluster_constrainshape')
        self._parser.add_argument('--upperbody_name', type=str, default='upperbody_combine_22sep_size126_biggershape_1_002_LRimageencoderX10_rate0.5_noisyZ100_withbetaaugmentation_11epochs')
        self._parser.add_argument('--upperbody_loadepoch', type=int, default=11)
        self._parser.add_argument('--upperbody_clusters', type=str, default='indexs_clusters_tshirt_smpl.npy')
        self._parser.add_argument('--upperbody_num_clusters', type=int, default=500)
        self._parser.add_argument('--upperbody_n_z_cut', type=int, default=6)
        self._parser.add_argument('--upperbody_n_z_style', type=int, default=12)
        self._parser.add_argument('--upperbody_resolution', type=int, default=128)
        self._parser.add_argument('--upperbody_thresh_occupancy', type=float, default=-0.055)

        self._parser.add_argument('--upperbody_highfreq_model', type=str, default='model_posedependent_deformation_0.001_shirts_reg')
        self._parser.add_argument('--upperbody_highfreq_name', type=str, default='model_posedependent_deformation_0.001_shirts_reg')
        self._parser.add_argument('--upperbody_highfreq_loadepoch', type=int, default=400000)

        self._parser.add_argument('--upperbody_donormals', type=bool, default=False)
        self._parser.add_argument('--upperbody_dodisplacement', type=bool, default=True)
        self._parser.add_argument('--upperbody_posedependentdisplacement', type=bool, default=True)

        # Pants options:
        self._parser.add_argument('--pants_model', type=str, default='bcnet_sdf_dist2eachcluster_constrainshape_lowerbody')
        self._parser.add_argument('--pants_name', type=str, default='refine_fulllowerbody_to_pants_0005_0005_0.04APosed')
        self._parser.add_argument('--pants_loadepoch', type=int, default=60)
        self._parser.add_argument('--pants_clusters', type=str, default='clusters_lowerbody.npy')
        self._parser.add_argument('--pants_num_clusters', type=int, default=500)
        self._parser.add_argument('--pants_n_z_cut', type=int, default=6)
        self._parser.add_argument('--pants_n_z_style', type=int, default=12)
        self._parser.add_argument('--pants_resolution', type=int, default=128)
        self._parser.add_argument('--pants_thresh_occupancy', type=float, default=-0.08)

        self._parser.add_argument('--pants_highfreq_model', type=str, default='bcnet_predictdisplacement_pants_posedependent')
        self._parser.add_argument('--pants_highfreq_name', type=str, default='model_posedependent_deformation_0.001_pants_reg')
        self._parser.add_argument('--pants_highfreq_loadepoch', type=int, default=400000)
        self._parser.add_argument('--pants_donormals', type=bool, default=False)
        self._parser.add_argument('--pants_dodisplacement', type=bool, default=True)
        self._parser.add_argument('--pants_posedependentdisplacement', type=bool, default=True)

        # Skirts options:
        self._parser.add_argument('--skirts_model', type=str, default='bcnet_sdf_dist2eachcluster_constrainshape_lowerbody')
        self._parser.add_argument('--skirts_name', type=str, default='lowerbody_debug_v4_6shape12style_01_01_0.04Aposed_40rate_addingnoise100_skirts')
        self._parser.add_argument('--skirts_loadepoch', type=int, default=40)
        self._parser.add_argument('--skirts_clusters', type=str, default='clusters_lowerbody.npy')
        self._parser.add_argument('--skirts_num_clusters', type=int, default=500)
        self._parser.add_argument('--skirts_n_z_cut', type=int, default=6)
        self._parser.add_argument('--skirts_n_z_style', type=int, default=12)
        self._parser.add_argument('--skirts_resolution', type=int, default=128)
        self._parser.add_argument('--skirts_thresh_occupancy', type=float, default=-0.05)

        self._parser.add_argument('--skirts_highfreq_model', type=str, default='bcnet_predictnormals_lowerbody_imgencoding')
        self._parser.add_argument('--skirts_highfreq_name', type=str, default='lowerbody_normals_exagerated')
        self._parser.add_argument('--skirts_highfreq_loadepoch', type=int, default=200000)
        self._parser.add_argument('--skirts_donormals', type=bool, default=True)
        self._parser.add_argument('--skirts_dodisplacement', type=bool, default=False)
        self._parser.add_argument('--skirts_posedependentdisplacement', type=bool, default=False)

        # Hair options:
        self._parser.add_argument('--hair_model', type=str, default='hair_sdf_dist2eachcluster_constrainshape_addingnoiseZ2')
        self._parser.add_argument('--hair_name', type=str, default='hairs_v4_6oct_126_01_0001_addingnoise100_prova')
        self._parser.add_argument('--hair_loadepoch', type=int, default=20000)
        self._parser.add_argument('--hair_clusters', type=str, default='clusters_hairs.npy')
        self._parser.add_argument('--hair_num_clusters', type=int, default=500)
        self._parser.add_argument('--hair_n_z_cut', type=int, default=6)
        self._parser.add_argument('--hair_n_z_style', type=int, default=12)
        self._parser.add_argument('--hair_resolution', type=int, default=512)
        self._parser.add_argument('--hair_thresh_occupancy', type=float, default=-2.0)
        #self._parser.add_argument('--hair_thresh_occupancy', type=float, default=-1.8)

        self._parser.add_argument('--hair_highfreq_model', type=str, default='bcnet_predictdisplacement_hair_imgencoding')
        self._parser.add_argument('--hair_highfreq_name', type=str, default='hairs_predictdisplacement_overfit12')
        self._parser.add_argument('--hair_highfreq_loadepoch', type=int, default=20000)
        self._parser.add_argument('--hair_donormals', type=bool, default=False)
        self._parser.add_argument('--hair_dodisplacement', type=bool, default=True)
        self._parser.add_argument('--hair_posedependentdisplacement', type=bool, default=False)

        # Shoes options
        self._parser.add_argument('--shoes_model', type=str, default='shoes_sdf_dist2eachcluster_constrainshape_addingnoiseZ')
        self._parser.add_argument('--shoes_name', type=str, default='shoes_11oct_size4')
        self._parser.add_argument('--shoes_loadepoch', type=int, default=20000)
        self._parser.add_argument('--shoes_clusters', type=str, default='clusters_shoes.npy')
        self._parser.add_argument('--shoes_n_z_cut', type=int, default=0)
        self._parser.add_argument('--shoes_n_z_style', type=int, default=4)
        self._parser.add_argument('--shoes_resolution', type=int, default=128)
        self._parser.add_argument('--shoes_thresh_occupancy', type=float, default=-0.04)

        self._parser.add_argument('--shoes_num_clusters', type=int, default=100)
        self._parser.add_argument('--shoes_donormals', type=bool, default=False)
        self._parser.add_argument('--shoes_dodisplacement', type=bool, default=False)
        self._parser.add_argument('--shoes_posedependentdisplacement', type=bool, default=False)

        # General options:
        # TODO ADD ANY? OF SMPL / PATHS TO CHECKPOINT FOLDER / ...
        #self._parser.add_argument('--path_checkpoints', type=str, default='/home/enric/cvpr21/model/checkpoints/')
        self._parser.add_argument('--path_checkpoints', type=str, default='../checkpoints/')
        self._parser.add_argument('--path_cluster_files', type=str, default='../clusters/')
        self._parser.add_argument('--path_SMPL', type=str, default='utils/neutral_smpl_with_cocoplus_reg.txt')
        #self._parser.add_argument('--path_SMPL', type=str, default='/media/enric/DATA/cvpr21/data_bcnet/BCNet/smpl_pytorch/model/neutral_smpl_with_cocoplus_reg.txt')

    def parse(self):
        self.initialize()
        self._opt = self._parser.parse_args()

        self._opt.upperbody_b_min = [-0.8, -0.4, -0.3]
        self._opt.upperbody_b_max = [0.8, 0.6, 0.3]
        self._opt.pants_b_min = [-0.3, -1.2, -0.3]
        self._opt.pants_b_max = [0.3, 0.0, 0.3]
        self._opt.skirts_b_min = [-0.3, -1.2, -0.3]
        self._opt.skirts_b_max = [0.3, 0.0, 0.3]
        self._opt.hair_b_min = [-0.35, -0.42, -0.33]
        self._opt.hair_b_max = [0.35, 0.68, 0.37]
        self._opt.shoes_b_min = [-0.1, -1.4, -0.2]
        self._opt.shoes_b_max = [0.25, -0.6, 0.3]

        return self._opt
