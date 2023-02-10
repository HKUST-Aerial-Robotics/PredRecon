import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('.')
from ops.voxel_module import Voxelization
from ops.voxel_vis import *
from einops import rearrange

class PillarLayer(nn.Module):
    """
    PointPillars: Fast Encoders for Object Detection from Point Clouds
    (https://arxiv.org/pdf/1812.05784.pdf)
    """
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        super().__init__()
        self.voxel_layer = Voxelization(voxel_size=voxel_size,
                                        point_cloud_range=point_cloud_range,
                                        max_num_points=max_num_points,
                                        max_voxels=max_voxels)

    @torch.no_grad()
    def forward(self, batched_pts):
        '''
        batched_pts: list[tensor], len(batched_pts) = bs
        return: 
               pillars: (p1 + p2 + ... + pb, num_points, c), 
               coors_batch: (p1 + p2 + ... + pb, 1 + 3), 
               num_points_per_pillar: (p1 + p2 + ... + pb, ), (b: batch size)
        '''
        pillars, coors, npoints_per_pillar = [], [], []
        for i, pts in enumerate(batched_pts):
            voxels_out, coors_out, num_points_per_voxel_out = self.voxel_layer(pts) 
            # voxels_out: (max_voxel, num_points, c), coors_out: (max_voxel, 3)
            # num_points_per_voxel_out: (max_voxel, )
            pillars.append(voxels_out)
            coors.append(coors_out.long())
            npoints_per_pillar.append(num_points_per_voxel_out)
        
        pillars = torch.cat(pillars, dim=0) # (p1 + p2 + ... + pb, num_points, c)
        npoints_per_pillar = torch.cat(npoints_per_pillar, dim=0) # (p1 + p2 + ... + pb, )
        coors_batch = []
        for i, cur_coors in enumerate(coors):
            coors_batch.append(F.pad(cur_coors, (1, 0), value=i))
        coors_batch = torch.cat(coors_batch, dim=0) # (p1 + p2 + ... + pb, 1 + 3)

        return pillars, coors_batch, npoints_per_pillar

class PointPillarsEncoder(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, in_channel, out_channel, input_channel, out_channels,deconv_channel,upsample_strides):
        super().__init__()
        self.out_channel = out_channel
        self.vx, self.vy = voxel_size[0], voxel_size[1]
        self.x_offset = voxel_size[0] / 2 + point_cloud_range[0]
        self.y_offset = voxel_size[1] / 2 + point_cloud_range[1]
        self.x_l = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
        self.y_l = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])

        self.conv = nn.Conv1d(in_channel, out_channel, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)
        
        # pseudo image feature extractor backbone
        # FPN
        self.multi_blocks = nn.ModuleList()
        layer_nums = [1, 1, 1]
        layer_strides=[2, 2, 2]
        for i in range(len(layer_strides)):
            blocks = []
            blocks.append(nn.Conv2d(input_channel, out_channels[i], 3, stride=layer_strides[i], bias=False, padding=1))
            blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
            blocks.append(nn.ReLU(inplace=True))

            for _ in range(layer_nums[i]):
                blocks.append(nn.Conv2d(out_channels[i], out_channels[i], 3, bias=False, padding=1))
                blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
                blocks.append(nn.ReLU(inplace=True))

            input_channel = out_channels[i]
            self.multi_blocks.append(nn.Sequential(*blocks))
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # Deconv
        self.deconv_blocks = nn.ModuleList()
        for i in range(len(out_channels)):
            de_block = []
            de_block.append(nn.ConvTranspose2d(out_channels[i], 
                                                    deconv_channel, 
                                                    upsample_strides[i], 
                                                    stride=upsample_strides[i],
                                                    bias=False))
            de_block.append(nn.BatchNorm2d(deconv_channel, eps=1e-3, momentum=0.01))
            de_block.append(nn.ReLU(inplace=True))

            self.deconv_blocks.append(nn.Sequential(*de_block))

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    
    def forward(self, pillars, coors_batch, npoints_per_pillar):
        '''
        pillars: (pillars_1+pillars_2+...+pillars_bs, num_points, c), c = 3 [x,y,z]
        coors_batch: (pillars_1+pillars_2+...+pillars_bs, 1 + 3), indexer
        npoints_per_pillar: (pillars_1+pillars_2+...+pillars_bs, ), number of points in each pillar
        return:  [pseudo image](bs, out_channel, y_l, x_l), [features]list
        '''
        device = pillars.device
        # 1. Calculate offset of points to the points center (in each pillar)
        # (pillars_1+pillars_2+...+pillars_bs, num_points, 3)
        offset_pt_center = pillars[:, :, :3] - torch.sum(pillars[:, :, :3], dim=1, keepdim=True) / npoints_per_pillar[:, None, None]

        # 2. Calculate offset of points to the pillar center (in each pillar) --- pseudo image
        x_offset_pi_center = pillars[:, :, :1] - (coors_batch[:, None, 1:2] * self.vx + self.x_offset) # (pillars_1+pillars_2+...+pillars_bs, num_points, 1)
        y_offset_pi_center = pillars[:, :, 1:2] - (coors_batch[:, None, 2:3] * self.vy + self.y_offset) # (pillars_1+pillars_2+...+pillars_bs, num_points, 1)
        
        # 3. Encoding
        features = torch.cat([pillars, offset_pt_center, x_offset_pi_center, y_offset_pi_center], dim=-1)

        # 4. Find mask for (0, 0, 0) and update the encoded features
        voxel_ids = torch.arange(0, pillars.size(1)).to(device) # (num_points, )
        mask = voxel_ids[:, None] < npoints_per_pillar[None, :] # (num_points, pillars_1+pillars_2+...+pillars_bs)
        mask = mask.permute(1, 0).contiguous()  # (pillars_1+pillars_2+...+pillars_bs, num_points)
        features *= mask[:, :, None]

        # 5. Embedding
        features = features.permute(0, 2, 1).contiguous() # (pillars_1+pillars_2+...+pillars_bs, 8, num_points)
        features = F.relu(self.bn(self.conv(features)))  # (pillars_1+pillars_2+...+pillars_bs, out_channels, num_points)
        pooling_features = torch.max(features, dim=-1)[0] # (pillars_1+pillars_2+...+pillars_bs, out_channels)

        # 6. Pillar scatter
        batched_canvas = []
        bs = coors_batch[-1, 0] + 1
        for i in range(bs):
            cur_coors_idx = coors_batch[:, 0] == i
            cur_coors = coors_batch[cur_coors_idx, :]
            cur_features = pooling_features[cur_coors_idx]

            canvas = torch.zeros((self.x_l, self.y_l, self.out_channel), dtype=torch.float32, device=device)
            canvas[cur_coors[:, 1], cur_coors[:, 2]] = cur_features
            canvas = canvas.permute(2, 1, 0).contiguous()
            batched_canvas.append(canvas)
        batched_canvas = torch.stack(batched_canvas, dim=0) # (bs, out_channel, self.y_l, self.x_l)
        x = batched_canvas
        
        # 7. Hierarchical scale features
        outs = []
        for i in range(len(self.multi_blocks)):
            x = self.multi_blocks[i](x)
            outs.append(x)
        
        upsample = []
        for i in range(len(self.deconv_blocks)):
            xi = self.deconv_blocks[i](outs[i])
            xi_r = rearrange(xi, 'b d h w -> b d (h w)')
            upsample.append(xi_r)
        out = torch.cat(upsample, dim=-1)
        
        # return batched_canvas, out
        return out

class PCNDecoder(nn.Module):
    """
    PCN: Point Cloud Completion Network
    (https://arxiv.org/pdf/1808.00671.pdf)
    Decoder Reference
    """
    def __init__(self, input_channel, pillarsize, num_dense=16384, input_num=8192, latent_dim=512, grid_size=1):
        super().__init__()
        
        self.in_channel = input_channel
        self.num_dense = num_dense - input_num
        self.latent_dim = latent_dim
        self.grid_size = grid_size
        self.pillarsize = pillarsize

        self.num_coarse = self.num_dense // (self.grid_size ** 2)

        self.mlp_repara = nn.Sequential(
            nn.Linear(self.pillarsize, self.num_coarse),
            nn.ReLU(inplace=True)
        )

        self.mlp_repara_2 = nn.Sequential(
            nn.Linear(self.in_channel, self.latent_dim),
            nn.ReLU(inplace=True)
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_dim, 3)
        )

        self.final_conv = nn.Sequential(
            nn.Conv1d(self.latent_dim + 3 + 2, int(self.latent_dim/2), 1),
            nn.BatchNorm1d(int(self.latent_dim/2)),
            nn.ReLU(inplace=True),
            nn.Conv1d(int(self.latent_dim/2), int(self.latent_dim/2), 1),
            nn.BatchNorm1d(int(self.latent_dim/2)),
            nn.ReLU(inplace=True),
            nn.Conv1d(int(self.latent_dim/2), 3, 1)
        )
        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
        
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2).cuda()  # (1, 2, S)
    
    def forward(self, xyz, pillar_feature):
        '''
        :xyz: (B, 8192, 3)
        :pillar_feature: (B, D, Pillarsize)
        :return: Predicted Point Clouds
        '''
        B, _, _ = pillar_feature.shape

        pillar_repara = self.mlp_repara(pillar_feature)
        feature_global = rearrange(pillar_repara, 'b d n -> b n d')
        feature_global = self.mlp_repara_2(feature_global)
        distill_feature = feature_global

        coarse = self.mlp(feature_global)
        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)             # (B, num_coarse, S, 3)
        point_feat = point_feat.reshape(-1, self.num_dense, 3).transpose(2, 1)               # (B, 3, num_fine)

        seed = self.folding_seed.unsqueeze(2).expand(B, -1, self.num_coarse, -1).to(xyz.device)             # (B, 2, num_coarse, S)
        seed = seed.reshape(B, -1, self.num_dense)                                           # (B, 2, num_fine)

        feature_global = rearrange(feature_global, 'b n d -> b d n')                         # (B, 1024, num_fine)
        feat = torch.cat([feature_global, seed, point_feat], dim=1)                          # (B, 1024+2+3, num_fine)
    
        fine = self.final_conv(feat) + point_feat                                            # (B, 3, num_fine), fine point cloud
        
        coarse = torch.cat((xyz, coarse), dim=1)                                             # (B, num_coarse, 3), coarse point cloud
        pred = fine.transpose(1, 2)
        fine = torch.cat((xyz, fine.transpose(1, 2)), dim=1)                                 # (B, num_dense, 3), fine point cloud

        return coarse.contiguous(), fine.contiguous(), pred.contiguous(), distill_feature.contiguous()

class SPM(nn.Module):
    """
    PredRecon --> Surface Prediction Module
    """
    def __init__(self,voxel_size, point_cloud_range, max_num_pt, max_voxels, in_channel, out_channel, input_channel, out_channels, deconv_channel, upsample_strides,
    decoder_inchannel, pillarsize, num_dense=16384, input_num=8192, latent_dim=512, grid_size=1):
        super().__init__()
        self.Pillars = PillarLayer(voxel_size, point_cloud_range, max_num_pt, max_voxels)
        self.Encoder = PointPillarsEncoder(voxel_size, point_cloud_range, in_channel, out_channel, input_channel, out_channels, deconv_channel, upsample_strides)
        self.Decoder = PCNDecoder(decoder_inchannel, pillarsize, num_dense, input_num, latent_dim, grid_size)
    
    def forward(self, xyz):
        '''
        :xyz: B, N, 3
        '''
        B,_,_ = xyz.shape
        bs_cloud = []
        for i in range(B):
            bs_cloud.append(xyz[i])

        pillars, coors, num = self.Pillars.forward(bs_cloud)
        out = self.Encoder.forward(pillars, coors, num)
        coarse, fine, pred, feature = self.Decoder.forward(xyz, out)

        return coarse, fine, pred, feature

class SPM_distill(nn.Module):
    """
    "PCN: Point Cloud Completion Network"
    (https://arxiv.org/pdf/1808.00671.pdf)

    Attributes:
        num_dense:  16384
        latent_dim: 512
        grid_size:  4
        num_coarse: 1024
    """

    def __init__(self, num_dense=16384, input_num=8192, latent_dim=512, grid_size=1):
        super().__init__()

        self.num_dense = num_dense - input_num
        self.latent_dim = latent_dim
        self.grid_size = grid_size

        assert self.num_dense % self.grid_size ** 2 == 0

        self.num_coarse = self.num_dense // (self.grid_size ** 2)

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
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_dim, 3 * self.num_coarse)
        )

        self.final_conv = nn.Sequential(
            nn.Conv1d(self.latent_dim + 3 + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 3, 1)
        )
        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
        
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2).cuda()  # (1, 2, S)

    def forward(self, xyz):
        B, N, _ = xyz.shape
        
        # encoder
        feature = self.first_conv(xyz.transpose(2, 1))                                       # (B,  256, N)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]                          # (B,  256, 1)
        feature = torch.cat([feature_global.expand(-1, -1, N), feature], dim=1)              # (B,  512, N)
        feature = self.second_conv(feature)                                                  # (B, 512, N)
        feature_global = torch.max(feature,dim=2,keepdim=False)[0]                           # (B, 512)
        
        # decoder
        coarse = self.mlp(feature_global).reshape(-1, self.num_coarse, 3)                    # (B, num_coarse, 3), coarse point cloud
        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)             # (B, num_coarse, S, 3)
        point_feat = point_feat.reshape(-1, self.num_dense, 3).transpose(2, 1)               # (B, 3, num_fine)

        seed = self.folding_seed.unsqueeze(2).expand(B, -1, self.num_coarse, -1).to(xyz.device)             # (B, 2, num_coarse, S)
        seed = seed.reshape(B, -1, self.num_dense)                                           # (B, 2, num_fine)

        feature_global = feature_global.unsqueeze(2).expand(-1, -1, self.num_dense)          # (B, 512, num_fine)
        distill_feature = feature_global.transpose(1,2)                                      # (B, num_fine, 512)
        feat = torch.cat([feature_global, seed, point_feat], dim=1)                          # (B, 512+2+3, num_fine)
    
        fine = self.final_conv(feat) + point_feat                                            # (B, 3, num_fine), fine point cloud
        
        coarse = torch.cat((xyz, coarse), dim=1)                                             # (B, num_coarse, 3), coarse point cloud
        pred = fine.transpose(1, 2)
        fine = torch.cat((xyz, fine.transpose(1, 2)), dim=1)                                 # (B, num_dense, 3), fine point cloud

        return coarse.contiguous(), fine.contiguous(), pred.contiguous(), distill_feature.contiguous()    # fine is the output pointcloud 


