import open3d as o3d
import numpy as np
import os

if __name__ == '__main__':
    batch = 0
    id = 0

    input_path = '/home/albert/dataset/Example/SceneExp/valid/partial'
    pred_path = '/home/albert/dataset/Example/SceneExp/prediction'
    gt_path = '/home/albert/dataset/Example/SceneExp/valid/complete'

    input_ = 'input_'+str(batch)+'_'+str(id)+'.pcd'
    pred_ = 'output_'+str(batch)+'_'+str(id)+'.pcd'
    gt_ = 'gt_'+str(batch)+'_'+str(id)+'.pcd'


    # axis_pcd = o3d.create_mesh_coordinate_frame(size=0.5, origin=[0, 0, 0])

    # ori = o3d.io.read_point_cloud(os.path.join(input_path, input_))
    # ori.paint_uniform_color([0, 0, 1.0])
    pred = o3d.io.read_point_cloud(os.path.join(pred_path))
    pred.paint_uniform_color([0, 1.0, 0])
    gt = o3d.io.read_point_cloud(os.path.join(gt_path))
    gt.paint_uniform_color([1.0, 0, 0])
    
    # o3d.visualization.draw_geometries([pred]+[axis_pcd])
    o3d.visualization.draw_geometries([gt, pred])
