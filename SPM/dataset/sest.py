import sys
import os
sys.path.append('.')

import os
import random

import torch
import torch.utils.data as data
import numpy as np
import open3d as o3d

class ScaleEstimate(data.Dataset):
    """
    ScaleEstimationHeader in SPM
    """
    
    def __init__(self, dataroot, split, category):
        assert split in ['train', 'valid', 'test', 'test_novel'], "split error value!"

        self.cat2id = {
            "house_train"  :  "0505",
            "house_test"  :  "0219",
            "house_valid"  :  "0405",
            "house_01"  :  "01",
            "house_02"  :  "02",
            "house_03"  :  "03",
            "house_04"  :  "04",
            "house_05"  :  "05",
            "house_06"  :  "06",
            "house_07"  :  "07",
            "OldRecon"    :  "0916"
        }

        self.dataroot = dataroot
        self.split = split
        self.category = category
        self.anchors = [10, 20, 30, 40, 50]

        self.partial_paths, self.complete_paths = self._load_data()
    
    def __getitem__(self, index):
        if self.split == 'train':
            partial_path = self.partial_paths[index]
        else:
            partial_path = self.partial_paths[index]
        complete_path = self.complete_paths[index]
        
        partial_list = partial_path.split('/')
        
        partial_pc_init = self.read_point_cloud(partial_path)

        partial_pc = self.random_sample(partial_pc_init, 8192) # patial_shape: (2048, 3)
        complete_pc = self.random_sample(self.read_point_cloud(complete_path), 16384) # complete_shape: (16384, 3)
        
        # normalization
        if partial_list[-2] == '0505' or partial_list[-2] == '0219' or partial_list[-2] == '0405':
            # data fix
            transform = np.array([[1,0,0],
                                  [0,0,1],
                                  [0,1,0]])
            # for start data
            # partial_pc = partial_pc.dot(transform)
            complete_pc = complete_pc.dot(transform)

            # # physical size
            partial_pc = partial_pc*self.scale
            complete_pc = complete_pc*self.scale

        # local transform
        center = np.mean(partial_pc, axis=0) 
        partial_pc[:,:3] = partial_pc[:,:3] - center[:3]
        complete_pc[:,:3] = complete_pc[:,:3] - center[:3]

        scale = self.scale_3axes(complete_pc)
        onehot_label = self.prob_label(scale)
        # ----------------------------
        return torch.from_numpy(partial_pc).type(torch.FloatTensor), scale, onehot_label

    def __len__(self):
        return len(self.complete_paths)

    def _load_data(self):
        with open(os.path.join(self.dataroot, '{}.list').format(self.split), 'r') as f:
            lines = f.read().splitlines()

        if self.category != 'all':
            lines = list(filter(lambda x: x.startswith(self.cat2id[self.category]), lines))
        
        partial_paths, complete_paths = list(), list()

        for line in lines:
            category, model_id = line.split('/')
            if self.split == 'train':
                # change .pcd file
                partial_paths.append(os.path.join(self.dataroot, self.split, 'partial', category, model_id + '.pcd'))
            else:
                partial_paths.append(os.path.join(self.dataroot, self.split, 'partial', category, model_id + '.pcd'))
            complete_paths.append(os.path.join(self.dataroot, self.split, 'complete', category, model_id + '.pcd'))
        
        return partial_paths, complete_paths
    
    def normalize(self, vec, bound):
        '''
        normalize input point cloud to 0-1
        '''
        vec[:,0] = vec[:,0]/bound
        vec[:,1] = vec[:,1]/bound
        vec[:,2] = vec[:,2]/bound

        return vec
    
    def read_point_cloud(self, path):
        pc = o3d.io.read_point_cloud(path)
        return np.array(pc.points, np.float32)
    
    def random_sample(self, pc, n):
        idx = np.random.permutation(pc.shape[0])
        if idx.shape[0] < n:
            idx = np.concatenate([idx, np.random.randint(pc.shape[0], size=n-pc.shape[0])])
        return pc[idx[:n]]
    
    def scale_3axes(self, comp):
        x_scale = np.abs(np.max(comp[:,0], axis=0)-np.min(comp[:,0], axis=0))
        y_scale = np.abs(np.max(comp[:,1], axis=0)-np.min(comp[:,1], axis=0))
        z_scale = np.abs(np.max(comp[:,2], axis=0)-np.min(comp[:,2], axis=0))

        scale = np.array([x_scale, y_scale, z_scale])
        scale_label = torch.from_numpy(scale)

        return scale_label
    
    def prob_label(self, scale_label):
        x_idx = 0
        y_idx = 0
        z_idx = 0
        x_res = 1000.0
        y_res = 1000.0
        z_res = 1000.0

        for i in range(len(self.anchors)):
            temp_res = self.anchors[i] - scale_label[0]
            if temp_res < x_res and temp_res > 0:
                x_idx = i
                x_res = temp_res
        
        for i in range(len(self.anchors)):
            temp_res = self.anchors[i] - scale_label[1]
            if temp_res < y_res and temp_res > 0:
                y_idx = i
                y_res = temp_res
        
        for i in range(len(self.anchors)):
            temp_res = self.anchors[i] - scale_label[2]
            if temp_res < z_res and temp_res > 0:
                z_idx = i
                z_res = temp_res
        
        x_list = [0 for i in range(len(self.anchors))]
        y_list = [0 for j in range(len(self.anchors))]
        z_list = [0 for k in range(len(self.anchors))]
        x_list[x_idx] = 1
        y_list[y_idx] = 1
        z_list[z_idx] = 1

        x_label = torch.from_numpy(np.array(x_list)).unsqueeze(0).to(dtype=torch.float32)
        y_label = torch.from_numpy(np.array(y_list)).unsqueeze(0).to(dtype=torch.float32)
        z_label = torch.from_numpy(np.array(z_list)).unsqueeze(0).to(dtype=torch.float32)

        labels = torch.cat((x_label, y_label, z_label), dim=0)

        return labels
