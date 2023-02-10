from einops.einops import rearrange
import torch
import torch.nn as nn
from torch.nn import functional
import numpy as np
import copy
from thop import profile
import sys
sys.path.append('.')

class ScaleEst(nn.Module):
    """
    Scale Estimator
    """
    def __init__(self, input_num = 8192, latent_dim = 512, anchors=[10, 20, 30, 40, 50]):
        super().__init__()
        self.partial_num = input_num
        self.latent_dim = latent_dim
        self.anchor_vec = torch.from_numpy(np.array(anchors)).unsqueeze(0).unsqueeze(0)
        self.headers = len(anchors)
        self.coors = 3 #xyz

        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.latent_dim, 1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_dim, self.coors)
        )

        self.prob_mlp = nn.Sequential(
            nn.Linear(self.latent_dim+self.coors, 128, True),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.headers, True)
        )

        self.reg_mlp = nn.Sequential(
            nn.Linear(self.latent_dim+self.coors, 128, True),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1, True)
        )
        
        self.local_mlp = nn.Linear(self.latent_dim+self.coors, 128, True)
        # self.local_mlp = nn.Linear(self.latent_dim, 128, True)

        self.offset_mlp = nn.Sequential(
            nn.Linear(128, 128, True),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1, True)
        )

        self.softmax = nn.Softmax(dim=-1)
    
    def positional_encoding_F_local(self, x):
        '''
        Positional embedding
        '''
        device = x.device
        onehots_ = functional.one_hot(torch.tensor(range(self.coors)), num_classes=self.coors).view(1,self.coors,1,self.coors)
        onehots_ = onehots_.repeat(x.shape[0], 1, self.partial_num, 1).to(device)
        x = torch.cat((x, onehots_.float()), dim=-1)

        return x
    
    def positional_encoding_F_global(self, x):
        '''
        Positional embedding
        '''
        device = x.device
        onehots_ = functional.one_hot(torch.tensor(range(self.coors)), num_classes=self.coors).view(1,self.coors,1,self.coors)
        onehots_ = onehots_.repeat(x.shape[0], 1, 1, 1).to(device)
        x = torch.cat((x, onehots_.float()), dim=-1)
        
        return x
    
    def forward(self, xyz):
        B, N, _ = xyz.shape
        
        device = xyz.device
        feature = self.first_conv(xyz.transpose(2, 1))                                       # (B,  256, N)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]                          # (B,  256, 1)
        feature = torch.cat([feature_global.expand(-1, -1, N), feature], dim=1)              # (B,  512, N)
        feature = self.second_conv(feature)                                                  # (B,  512, N)
        feature_local = rearrange(feature.unsqueeze(1), 'b c d n -> b c n d').repeat(1,self.coors,1,1)
        feature_local = self.positional_encoding_F_local(feature_local)                      # (B, 3, N, 512+3)
        feature_local = torch.mean(self.local_mlp(feature_local), dim=-2)                    # (B, 3, 128)

        feature_global = torch.max(feature,dim=2,keepdim=False)[0].unsqueeze(1).unsqueeze(1).repeat(1,self.coors,1,1)
                                                                                             # (B, 3, 1, 512)
        feature_global = self.positional_encoding_F_global(feature_global)

        # output
        reg_prob = self.softmax(self.prob_mlp(feature_global).squeeze(-2)).squeeze(-1)
        reg = self.reg_mlp(feature_global).squeeze(-1).squeeze(-1)
        offset = torch.tanh(self.offset_mlp(feature_local)).squeeze(-1)

        scale_result = reg + offset
        scale_prob = reg_prob

        return scale_result, scale_prob

if __name__ == '__main__':
    '''
    for test dimensions
    '''
    pcd = torch.randn(1, 8192, 3)
    scale_estimator = ScaleEst()
    reg, prob = scale_estimator.forward(pcd)
    print(reg.size())
    print(prob.size())
    label = functional.one_hot(torch.tensor(range(3)), num_classes=5).unsqueeze(0).to(dtype=torch.float32)

    # FLOPs and Params
    flops, params = profile(scale_estimator, inputs=(pcd,))
    print('ScaleEstimator_FLOPs:', flops/1e9,'GFLOPs.')
    print('ScaleEstimator_Params:', params/1e6,'M.')