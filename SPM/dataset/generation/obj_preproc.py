# -*- coding:UTF-8 -*-
# Used for processing and modifing obj files.
# This code only processes geometry information and ignores materials.

from hashlib import new
import os
import datetime
from tkinter.messagebox import NO
import numpy as np
 

class OBJ:
    def __init__(self, file_dir, file_name, material):
        '''
        Loads a Wavefront OBJ file.
        '''
        # (x, y, z)
        self.vertices = []
        # (x, y, z)
        self.after_v = []
        # (x, y, z)
        self.normals = []
        # (u, v)
        self.texcoords = []
        # (([v1, v2, ...], [vt1, vt2, ...], [vn1, vn2, ...]), ...)
        self.faces = []
        
        self.file_dir = file_dir
        self.file_name = file_name
        self.material = material

        for line in open(os.path.join(file_dir, file_name), "r"):
            # Comments
            if line.startswith('#'): 
                continue
            values = line.split()
            if not values: 
                continue
            # Vertex
            if values[0] == 'v':
                v = [float(i) for i in  values[1:4]]
                self.vertices.append(v)
            # Normal
            elif values[0] == 'vn':
                vn = [float(i) for i in values[1:4]]
                self.normals.append(vn)
            # UV coordinate
            elif values[0] == 'vt':
                vt = [float(i) for i in values[1:3]]
                self.texcoords.append(vt)
            # Face
            elif values[0] == 'f':
                face = []
                texcoord = []
                norm = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoord.append(int(w[1]))
                    else:
                        texcoord.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norm.append(int(w[2]))
                    else:
                        norm.append(0)
                self.faces.append([face, texcoord, norm])


    def export_obj(self, out_path, out_name=None, use_material=True):
        '''
        Export the obj model with new material.
        '''
        if out_name != None:
            out_path = os.path.join(out_path, out_name)
        else:    
            out_path = os.path.join(out_path, self.file_name)

        fout = open(out_path, "w")

        # Write time ans user info
        fout.write("# Wavefront obj\n")
        time_now = datetime.datetime.now().strftime("# Generated at %Y-%m-%d %H:%M:%S\n\n")
        fout.write(time_now)
  
        # Write material library
        if use_material == True:
            fout.write("mtllib " + self.material + ".mtl\n\n")

        # Write vertex
        # Format: "v x y z"
        for i in range(len(self.after_v)):
            v = self.after_v[i]
            v_str = "v " + str(v[0]) + " " + str(v[1]) + " " + str(v[2]) + "\n"
            fout.write(v_str)
        fout.write("# " + str(len(self.vertices)) + " elements written\n\n")
        
        # Write UV coordinate
        # Format: "vt u v"
        for i in range(len(self.texcoords)):
            vt = self.texcoords[i]
            vt_str = "vt " + str(vt[0]) + " " + str(vt[1])  + "\n"
            fout.write(vt_str)
        fout.write("# " + str(len(self.texcoords)) + " elements written\n\n")

        # Write vertex normal
        # Format: "vn x y z"
        for i in range(len(self.normals)):
            vn = self.normals[i]
            vn_str = "vn " + str(vn[0]) + " " + str(vn[1]) + " " + str(vn[2]) + "\n"
            fout.write(vn_str)
        fout.write("# " + str(len(self.normals)) + " elements written\n\n")

        # Write material
        # Format: "usemtl xxx"
        if use_material == True:
            fout.write("usemtl " + self.material + "\n\n")

        # Write face
        # Format: "f vi/vti/vni ..."
        for i in range(len(self.faces)):
            f = self.faces[i]
            f_str = "f"
            for j in range(len(f[0])):
                v_id = str(f[0][j])
                vt_id = str(f[1][j])
                vn_id = str(f[2][j])

                face_element = ""
                # Without uv and normal
                if vt_id == "0" and vn_id == "0":
                    face_element = " " + v_id
                # Without uv
                elif vt_id == "0" and vn_id != "0":
                    face_element = " " + v_id + "//" + vn_id
                # Without normal
                elif vt_id != "0" and vn_id == "0":
                    face_element = " " + v_id + "/" + vt_id
                elif vt_id != "0" and vn_id != "0":
                    face_element = " " + v_id + "/" + vt_id + "/" + vn_id

                f_str += face_element

            f_str += "\n"
            fout.write(f_str)
        fout.write("# " + str(len(self.faces)) + " elements written\n\n")

        fout.close()


    def scale_model(self, scale):
        '''
        Modify the scale of 3d model.
        '''
        pass


    def min_max_cal(self):
        '''
        Calculate bbox of an obj file.
        '''
        points = np.array(self.vertices).reshape((-1, 3))
        
        xyz_min = np.min(points, axis=0)
        xyz_max = np.max(points, axis=0)

        min_abs = np.abs(xyz_min)
        max_abs = np.abs(xyz_max)
        abs_cat = np.concatenate((min_abs, max_abs))
        scale = np.max(abs_cat)
        
        for i in self.vertices:
            temp = []
            for j in i:
                temp.append(j/scale)
            self.after_v.append(temp)

        return xyz_min, xyz_max, scale


if __name__ == '__main__':

    file_path = '/home/albert/dataset/house3K'
    proc_path = '/home/albert/dataset/house3K_norm'
    
    ori_obj_list = os.listdir(file_path)
    for i in ori_obj_list:
        if i.endswith('.obj'):
            temp_obj = OBJ(file_path, i, None)
            _,_,scale = temp_obj.min_max_cal()
            print(scale)
            new_name = i[:-4] +'_norm.obj'
            print(new_name)
            temp_obj.export_obj(proc_path, new_name, False)
