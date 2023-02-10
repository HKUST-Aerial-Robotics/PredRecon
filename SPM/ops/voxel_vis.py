import numpy
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def vis_input_pcd(pcd_array):
    x = pcd_array[:,0]
    y = pcd_array[:,1]
    z = pcd_array[:,2]

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z)

    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})

    plt.show()

def vis_voxelization(voxels):
    vox_num = voxels.shape[0]
     
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})

    for i in range(vox_num):
        x = voxels[i,:,0]
        y = voxels[i,:,1]
        z = voxels[i,:,2]
        ax.scatter(x, y, z)

    plt.show()
