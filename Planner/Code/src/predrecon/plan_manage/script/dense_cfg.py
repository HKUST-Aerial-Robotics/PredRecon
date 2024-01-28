import os

def set_dense_recon_cfg(folder, matching_num):
    imgs = os.path.join(folder, 'images')
    cfg = os.path.join(folder, 'stereo/patch-match.cfg')
    img_num = len(os.listdir(imgs))

    f = open(cfg, 'w')
    for i in range(1,img_num+1):
        idx = str(i).rjust(4,'0')
        img_name = 'image_' + idx + '.jpg'
        f.write(img_name+'\n')
        mvs_ids = distribute_MVS(i, img_num, matching_num)
        for j in range(len(mvs_ids)):
            id_m = 'image_' + str(mvs_ids[j]).rjust(4,'0') + '.jpg'
            f.write(id_m)
            if j != len(mvs_ids)-1:
                f.write(', ')
        f.write('\n')

def distribute_MVS(cur_id, max_num, frame):
    MVS_list = []
    if cur_id <= frame:
        for i in range(1,cur_id):
            MVS_list.append(i)
        for j in range(cur_id+1, 2*frame+2):
            MVS_list.append(j)
    
    elif cur_id > max_num-frame:
        for i in range(max_num-2*frame,cur_id):
            MVS_list.append(i)
        for j in range(cur_id+1, max_num+1):
            MVS_list.append(j)
    else:
        for i in range(cur_id-frame, cur_id):
            MVS_list.append(i)
        for j in range(cur_id+1, cur_id+frame+1):
            MVS_list.append(j)
    
    return MVS_list

if __name__ == '__main__':
    file = '/home/albert/UAV_Planning/Noise_test/imgs_wo_noise/dense'
    num = 20
    set_dense_recon_cfg(file, num)