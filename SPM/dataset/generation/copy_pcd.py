import shutil
import os

if __name__ == '__main__':
    count = 0
    for i in os.listdir('/home/albert/dataset/house3K_complete'):
        if i.endswith('_far.pcd') or i.endswith('_near.pcd'):
            count += 0
        else:
            print(i)
            count += 1
            temp_pcd = os.path.join('/home/albert/dataset/house3K_complete',i)
            shutil.copyfile(temp_pcd, os.path.join('/home/albert/dataset/house3K_complete',i[:-4]+'_nearest.pcd'))
    
    print('count:',count)