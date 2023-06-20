## Colmap Pipeline

* Folder Structure

```
├─created
│  └─sparse
│   +── cameras.txt
│   +── images.txt
│   +── points3D.txt
├─dense
├─images
│   +── images001.jpg
│   +── images002.jpg
│   +── ...
└─triangulated
    └─sparse
```

* Feature Extraction

```shell
colmap feature_extractor --database_path database.db --image_path images
```

* Update Intrinsic

```shell
python3 database.py --database_path /home/albert/UAV_Planning/Recon/planner_0831_2_spatial/database.db
```

* Feature Matching

```shell
colmap exhaustive_matcher (or other matcher in gui) --database_path database.db
```

* Mapper

```shell
colmap mapper --database_path database.db --image_path images --output_path triangulated/sparse
```

* Triangulation (Not necessary)

```shell
colmap point_triangulator --database_path database.db --image_path images --input_path created/sparse --output_path triangulated/sparse
```

* Dense Reconstruction

### If the sparse from point triangulation
```shell
colmap image_undistorter --image_path images --input_path triangulated/sparse --output_path dense
colmap patch_match_stereo --workspace_path dense
colmap stereo_fusion --workspace_path dense --output_path dense/fused.ply
```

### If the sparse from camera poses directly
```shell
colmap image_undistorter --image_path images --input_path created/sparse --output_path dense
python3 dense_cfg.py
colmap patch_match_stereo --workspace_path dense --PatchMatchStereo.depth_min 0.0 --PatchMatchStereo.depth_max 10.0
colmap stereo_fusion --workspace_path dense --output_path dense/fused.ply
```