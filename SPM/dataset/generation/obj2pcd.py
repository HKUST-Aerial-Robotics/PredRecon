import os

if __name__ == '__main__':
    result_path = '/home/albert/dataset/house3K_complete'
    file_path = '/home/albert/dataset/house3K'

    os.system('cd '+ file_path)
    print(len(os.listdir(file_path)))
    for i in os.listdir(file_path):
        if (i.endswith('.obj')):
            name = os.path.join(file_path,i)
            print(name)
            proc_name = os.path.join(result_path, i[:-4]+'.pcd')
            print(proc_name)
            os.system('pcl_mesh_sampling '+name+' '+proc_name+' -n_samples 200000 -leaf_size 0.002')