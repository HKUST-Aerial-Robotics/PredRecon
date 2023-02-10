from operator import mod
import numpy as np
import numpy.linalg
import cv2
import math
from numpy import array as matrix, arange
import imageio
import os
import open3d as o3d
import shutil

def generate_XYZ(depth, pose):
    r'''
    :T: T_mat
    '''
    x_rot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float32)
    z_rot = np.array([[0,-1,0],[1,0,0],[0,0,1]],np.float32)

    fx = 525.0
    fy = 525.0
    cx = 240.0
    cy = 240.0

    intrinsics = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]])

    inv_K = np.linalg.inv(intrinsics)
    inv_K[2, 2] = -1
    
    where_are_inf = np.isinf(depth)
    depth[where_are_inf] = -100.0

    x, y = np.where(depth > 0)
    # image coordinates -> camera coordinates
    points = np.dot(inv_K, np.stack([x, y, np.ones_like(x)] * depth[x, y], 0)).T
    pose_t = np.linalg.inv(pose)
    points = np.dot(np.linalg.inv(x_rot), np.dot(pose[:3,:3], np.dot(np.linalg.inv(z_rot), points.T) - pose_t[:3,3][:,np.newaxis])).T

    return points

def readimg(img_file, pose_file, pcd_name):

    img = imageio.imread(img_file)[...,0]
    res = img.shape

    depth = np.array(img)
    
    with open(pose_file, 'r') as f:
        content = f.read().splitlines()
    T_m_list = []
    for i in content:
        T_m_list.append(float(i))
    T_m = np.array(T_m_list).reshape(4,4)
    
    xyz = generate_XYZ(depth, T_m)
    print(xyz.shape)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    o3d.io.write_point_cloud(pcd_name, pcd, True)

    print('finish')

def generate_data(partial, complete, result, folder, mode='train'):
    
    partial_list = os.listdir(os.path.join(partial, mode))
    f_list = open(os.path.join(result, mode+'.list'), 'w')
    for i in partial_list:
        temp_complete = os.path.join(complete,i+'.pcd')
        num = len(os.listdir(os.path.join(partial,mode,i)))//5
        for j in range(num):
            ids = str(j).zfill(4)
            depth = os.path.join(partial,mode,i,str(j)+'_'+ids+'_L.exr')
            pose = os.path.join(partial,mode,i,str(j)+'.txt')

            readimg(depth, pose, os.path.join(result,mode,'partial/'+folder,i+'_'+str(j)+'.pcd'))
            shutil.copyfile(temp_complete, os.path.join(result,mode,'complete/'+folder,i+'_'+str(j)+'.pcd'))
            f_list.write(folder+'/'+i+'_'+str(j))
            f_list.write('\n')
            print('record finish!')
    
    f_list.close()
    

if __name__ == '__main__':
    # ---Example--- #
    file_path = '/home/albert/dataset/Example/SceneExp/valid/partial/1234'
    gt_pcd = '/home/albert/dataset/Example/gt_cloud.pcd'
    list_path = '/home/albert/dataset/Example/SceneExp/valid.list'
    complete_path = '/home/albert/dataset/Example/SceneExp/valid/complete/1234'

    f_list = open(list_path, 'w')
    for i in os.listdir(file_path):
        name = i[:-4]
        shutil.copyfile(os.path.join(gt_pcd), os.path.join(complete_path, i))
        f_list.write('1234/'+name)
        f_list.write('\n')
    
    f_list.close()