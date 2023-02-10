import bpy
import os

result_path = '/home/albert/dataset/house3K'

file_path = '/home/albert/dataset/raw_house_3K/FBX'
batch_file = 'Batch_12'

convert_path = os.path.join(file_path, batch_file)
counter = 272

for i in os.listdir(convert_path):
    fbx_list = os.listdir(os.path.join(convert_path,i))
    for j in fbx_list:
        if j.endswith(".fbx"):
            print(j)
            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.object.delete()

            bpy.ops.import_scene.fbx(filepath=os.path.join(convert_path,i,j))

            n = str(counter)
            s = n.zfill(2)
            obj_name = 'house_'+s+'.obj'
            bpy.ops.export_scene.obj(filepath=os.path.join(result_path, obj_name))
            counter += 1

print(counter)