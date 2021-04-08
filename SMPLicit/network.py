import torch.nn as nn
import numpy as np
import torchvision
import torch
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, n_z_style=1, point_pos_size=3, output_dim=1, n_z_cut=12):
        super(Network, self).__init__()
        self.point_pos_size = point_pos_size

        self.fc0_cloth = nn.utils.weight_norm(nn.Linear(n_z_style, 128, bias=True))
        self.fc1_cloth = nn.utils.weight_norm(nn.Linear(128, 128, bias=True))

        self.fc0_query = nn.utils.weight_norm(nn.Conv1d(point_pos_size, 128, kernel_size=1, bias=True))
        self.fc1_query = nn.utils.weight_norm(nn.Conv1d(128, 256, kernel_size=1, bias=True))

        self.fc0 = nn.utils.weight_norm(nn.Conv1d(128+256 + n_z_cut, 312, kernel_size=1, bias=True))
        self.fc1 = nn.utils.weight_norm(nn.Conv1d(312, 312, kernel_size=1, bias=True))
        self.fc2 = nn.utils.weight_norm(nn.Conv1d(312, 256, kernel_size=1, bias=True))
        self.fc3 = nn.utils.weight_norm(nn.Conv1d(256, 128, kernel_size=1, bias=True))
        self.fc4 = nn.utils.weight_norm(nn.Conv1d(128, output_dim, kernel_size=1, bias=True))

        self.activation = F.relu
        #self.activation = torch.sin

    def forward(self, uvmap_representation, cloth_representation, point_position):
        _B = len(cloth_representation)
        _numpoints = len(point_position[0])

        point_encoding = point_position
        point_encoding = point_encoding.reshape(_B, _numpoints, self.point_pos_size).permute(0,2,1)
        
        x_cloth = self.activation(self.fc0_cloth(cloth_representation))
        x_cloth = self.activation(self.fc1_cloth(x_cloth))

        x_position = self.activation(self.fc0_query(point_encoding))
        x_position = self.activation(self.fc1_query(x_position))

        uvmap_representation = uvmap_representation.unsqueeze(-1).repeat(1, 1, _numpoints)
        x_cloth = x_cloth.unsqueeze(1).repeat(1, _numpoints, 1).permute(0, 2, 1)
        _in = torch.cat((x_cloth, x_position, uvmap_representation), 1)

        x = self.fc0(_in)
        x = self.activation(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)
        x = self.fc4(x)

        if x.shape[1] == 1:
            return x[:, 0]
        else:
            return x
