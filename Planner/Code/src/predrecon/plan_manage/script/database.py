import sys
import numpy as np
import sqlite3

# CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
#     image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
#     name TEXT NOT NULL UNIQUE,
#     camera_id INTEGER NOT NULL,
#     prior_qw REAL,
#     prior_qx REAL,
#     prior_qy REAL,
#     prior_qz REAL,
#     prior_tx REAL,
#     prior_ty REAL,
#     prior_tz REAL,
#     CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
#     FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
# """.format(MAX_IMAGE_ID)

IS_PYTHON3 = sys.version_info[0] >= 3

def array_to_blob(array):
    if IS_PYTHON3:
        return array.tobytes()
    else:
        return np.getbuffer(array)

def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)

class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table =             lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table =             lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table =             lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table =             lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table =             lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table =             lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def update_camera(self, model, width, height, params, camera_id):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=True WHERE camera_id=?",
            (model, width, height, array_to_blob(params),camera_id))
        return cursor.lastrowid
    
    def add_image(self, image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name):
        cursor = self.execute("UPDATE images SET name=?, camera_id=?, prior_qw=?, prior_qx=?, prior_qy=?, prior_qz=?, prior_tx=?, prior_ty=?, prior_tz=? WHERE image_id=?",
            (name, camera_id, qw, qx, qy, qz, tx, ty, tz, image_id))
        return cursor.lastrowid

def camTodatabase(txtfile):
    import os
    import argparse

    camModelDict = {'SIMPLE_PINHOLE': 0,
                    'PINHOLE': 1,
                    'SIMPLE_RADIAL': 2,
                    'RADIAL': 3,
                    'OPENCV': 4,
                    'FULL_OPENCV': 5,
                    'SIMPLE_RADIAL_FISHEYE': 6,
                    'RADIAL_FISHEYE': 7,
                    'OPENCV_FISHEYE': 8,
                    'FOV': 9,
                    'THIN_PRISM_FISHEYE': 10}
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", default="database.db")
    args = parser.parse_args()
    if os.path.exists(args.database_path)==False:
        print("ERROR: database path dosen't exist -- please check database.db.")
        return
    # Open the database.
    db = COLMAPDatabase.connect(args.database_path)

    idList=list()
    modelList=list()
    widthList=list()
    heightList=list()
    paramsList=list()
    # Update real cameras from .txt
    with open(txtfile, "r") as cam:
        lines = cam.readlines()
        for i in range(0,len(lines),1):
            if lines[i][0]!='#':
                strLists = lines[i].split()
                cameraId=int(strLists[0])
                cameraModel=camModelDict[strLists[1]] #SelectCameraModel
                width=int(strLists[2])
                height=int(strLists[3])
                paramstr=np.array(strLists[4:12])
                params = paramstr.astype(np.float64)
                idList.append(cameraId)
                modelList.append(cameraModel)
                widthList.append(width)
                heightList.append(height)
                paramsList.append(params)
                camera_id = db.update_camera(cameraModel, width, height, params, cameraId)

    # Commit the data to the file.
    db.commit()
    # Read and check cameras.
    rows = db.execute("SELECT * FROM cameras")
    for i in range(0,len(idList),1):
        camera_id, model, width, height, params, prior = next(rows)
        params = blob_to_array(params, np.float64)
        assert camera_id == idList[i]
        assert model == modelList[i] and width == widthList[i] and height == heightList[i]
        assert np.allclose(params, paramsList[i])

    # Close database.db.
    db.close()

def imgTodatabase(txtfile):
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", default="database.db")
    args = parser.parse_args()
    if os.path.exists(args.database_path)==False:
        print("ERROR: database path dosen't exist -- please check database.db.")
        return
    # Open the database.
    db = COLMAPDatabase.connect(args.database_path)

    img_idList=list()
    nameList=list()
    cam_idList=list()
    qwList=list()
    qxList=list()
    qyList=list()
    qzList=list()
    txList=list()
    tyList=list()
    tzList=list()
    # Update real cameras from .txt
    with open(txtfile, "r") as cam:
        lines = cam.readlines()
        for i in range(0,len(lines),1):
            if lines[i][0]!='#' and len(lines[i]) > 1:
                strLists = lines[i].split()
                imgId = int(strLists[0])
                qw = float(strLists[1])
                qx = float(strLists[2])
                qy = float(strLists[3])
                qz = float(strLists[4])
                tx = float(strLists[5])
                ty = float(strLists[6])
                tz = float(strLists[7])
                # camId = int(strLists[8])
                camId = int(strLists[0])
                name = strLists[9]
                img_idList.append(imgId)
                nameList.append(name)
                cam_idList.append(camId)
                qwList.append(qw)
                qxList.append(qx)
                qyList.append(qy)
                qzList.append(qz)
                txList.append(tx)
                tyList.append(ty)
                tzList.append(tz)
                image_id = db.add_image(imgId, qw, qx, qy, qz, tx, ty, tz, camId, name)

    # Commit the data to the file.
    db.commit()
    # Read and check cameras.
    # rows = db.execute("SELECT * FROM cameras")
    # for i in range(0,len(img_idList),1):
    #     camera_id, model, width, height, params, prior = next(rows)
    #     assert camera_id == idList[i]
    #     assert model == modelList[i] and width == widthList[i] and height == heightList[i]
    #     assert np.allclose(params, paramsList[i])

    # Close database.db.
    db.close()

if __name__ == "__main__":
    file_path = '/home/albert/UAV_Planning/icra_recon/proposed_0/'
    imgTodatabase(file_path + "created/sparse/images.txt")