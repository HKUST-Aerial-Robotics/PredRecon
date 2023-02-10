# A simple script that uses blender to render views of a single object by rotation the camera around it.
# Also produces depth map at the same time.
#
# Example:
# blender --background --python mytest.py -- --views 10 /path/to/my.obj
#

import argparse, sys, os
import math
import mathutils
import numpy as np
parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument('--views', type=int, default=30,
                    help='number of views to be rendered')
parser.add_argument('obj', type=str,
                    help='Path to the obj file to be rendered.')
parser.add_argument('--output_folder', type=str, default='./test',
                    help='The path the output will be dumped to.')
parser.add_argument('--scale', type=float, default=1,
                    help='Scaling factor applied to model. Depends on size of mesh.')
parser.add_argument('--remove_doubles', type=bool, default=True,
                    help='Remove double vertices to improve mesh quality.')
parser.add_argument('--edge_split', type=bool, default=True,
                    help='Adds edge split filter.')
parser.add_argument('--depth_scale', type=float, default=1.4,
                    help='Scaling that is applied to depth. Depends on size of mesh. Try out various values until you get a good result. Ignored if format is OPEN_EXR.')
parser.add_argument('--color_depth', type=str, default='16',
                    help='Number of bit per channel used for output. Either 8 or 16.')
parser.add_argument('--format', type=str, default='OPEN_EXR',
                    help='Format of files generated. Either PNG or OPEN_EXR')

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)

import bpy

# Set up rendering of depth map.
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links

# Add passes for additionally dumping albedo and normals.
bpy.context.scene.render.layers["RenderLayer"].use_pass_normal = True
bpy.context.scene.render.layers["RenderLayer"].use_pass_color = True
bpy.context.scene.render.image_settings.file_format = args.format
bpy.context.scene.render.image_settings.color_depth = args.color_depth

# Clear default nodes
for n in tree.nodes:
    tree.nodes.remove(n)

# Create input render layer node.
render_layers = tree.nodes.new('CompositorNodeRLayers')

depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
depth_file_output.label = 'Depth Output'
if args.format == 'OPEN_EXR':
  links.new(render_layers.outputs['Z'], depth_file_output.inputs[0])
else:
  # Remap as other types can not represent the full range of depth.
  map = tree.nodes.new(type="CompositorNodeMapValue")
  # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
  map.offset = [-0.7]
  map.size = [args.depth_scale]
  map.use_min = True
  map.min = [0]
  links.new(render_layers.outputs['Z'], map.inputs[0])
  links.new(map.outputs[0], depth_file_output.inputs[0])

# scale_normal = tree.nodes.new(type="CompositorNodeMixRGB")
# scale_normal.blend_type = 'MULTIPLY'
# # scale_normal.use_alpha = True
# scale_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
# links.new(render_layers.outputs['Normal'], scale_normal.inputs[1])

# bias_normal = tree.nodes.new(type="CompositorNodeMixRGB")
# bias_normal.blend_type = 'ADD'
# # bias_normal.use_alpha = True
# bias_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
# links.new(scale_normal.outputs[0], bias_normal.inputs[1])

# normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
# normal_file_output.label = 'Normal Output'
# links.new(bias_normal.outputs[0], normal_file_output.inputs[0])

albedo_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
albedo_file_output.label = 'Albedo Output'
links.new(render_layers.outputs['Color'], albedo_file_output.inputs[0])

# Delete default cube
bpy.data.objects['Cube'].select = True
bpy.ops.object.delete()

bpy.ops.import_scene.obj(filepath=args.obj)

# for object in bpy.context.scene.objects:
#     if object.name in ['Camera', 'Lamp']:
#         continue
#     bpy.context.scene.objects.active = object
#     if args.scale != 1:
#         bpy.ops.transform.resize(value=(args.scale,args.scale,args.scale))
#         bpy.ops.object.transform_apply(scale=True)
#     if args.remove_doubles:
#         bpy.ops.object.mode_set(mode='EDIT')
#         bpy.ops.mesh.remove_doubles()
#         bpy.ops.object.mode_set(mode='OBJECT')
#     if args.edge_split:
#         bpy.ops.object.modifier_add(type='EDGE_SPLIT')
#         bpy.context.object.modifiers["EdgeSplit"].split_angle = 1.32645
#         bpy.ops.object.modifier_apply(apply_as='DATA', modifier="EdgeSplit")

from math import radians
bpy.context.scene.render.engine = 'BLENDER_GAME'
# bpy.context.scene.world.use_nodes = True
# bpy.context.scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = 5.0

# Make light just directional, disable shadows.
lamp = bpy.data.lamps['Lamp']
lamp.type = 'SUN'
lamp.shadow_method = 'NOSHADOW'
lamp.energy = 0.5
# Possibly disable specular shading:
lamp.use_specular = False
import random
score = random.randint(0,2)
bpy.data.objects['Lamp'].rotation_euler[0] = radians(135 + score * 45)
# Add another light source so stuff facing away from light is not completely dark
bpy.ops.object.lamp_add(type='SUN')
lamp2 = bpy.data.lamps['Sun']
lamp.shadow_method = 'NOSHADOW'
lamp2.use_specular = False
lamp2.energy = 0.15
bpy.data.objects['Sun'].rotation_euler = bpy.data.objects['Lamp'].rotation_euler
bpy.data.objects['Sun'].rotation_euler[0] += radians(180)

# print(bpy.context.scene.world.node_tree.nodes['Background'].inputs[0].default_value[:3])
# assert(0)
def parent_obj_to_camera(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.objects.link(b_empty)
    scn.objects.active = b_empty
    return b_empty

scene = bpy.context.scene
scene.render.resolution_x = 480
scene.render.resolution_y = 480
scene.render.resolution_percentage = 100
scene.render.alpha_mode = 'TRANSPARENT'

cam = scene.objects['Camera']
# y_random = 0.2+0.2*random.random()
y_random = 0.3
cam.location = (0.1, y_random, 0.2)
cam.data.stereo.interocular_distance = 0.065
cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'
b_empty = parent_obj_to_camera(cam)
cam_constraint.target = b_empty
scene.render.views_format = 'STEREO_3D'


# sensor_width_in_mm = cam.data.sensor_width
# sensor_height_in_mm = cam.data.sensor_height
# pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
# if (cam.data.sensor_fit == 'VERTICAL'):
#     # the sensor height is fixed (sensor fit is horizontal), 
#     # the sensor width is effectively changed with the pixel aspect ratio
#     s_u = scene.render.resolution_x / sensor_width_in_mm / pixel_aspect_ratio 
#     s_v = scene.render.resolution_y / sensor_height_in_mm
# else: # 'HORIZONTAL' and 'AUTO'
#     # the sensor width is fixed (sensor fit is horizontal), 
#     # the sensor height is effectively changed with the pixel aspect ratio
#     pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
#     s_u = scene.render.resolution_x / sensor_width_in_mm
#     s_v = scene.render.resolution_y * pixel_aspect_ratio / sensor_height_in_mm

# def setup_blender(width, height, focal_length):
#     # camera
#     camera = bpy.data.objects['Camera']
#     camera.data.angle = 0.8575560548920328

#     # render layer
#     scene = bpy.context.scene
#     scene.render.filepath = 'buffer'
#     scene.render.image_settings.color_depth = '16'
#     scene.render.resolution_percentage = 100
#     scene.render.resolution_x = width
#     scene.render.resolution_y = height

#     return camera
# # camm = setup_blender(160, 120, 100)
# f_in_mm = cam.data.lens
# alpha_u = f_in_mm * s_u
# alpha_v = f_in_mm * s_v
# u_0 = scene.render.resolution_x / 2
# v_0 = scene.render.resolution_y / 2
# print(f_in_mm, sensor_width_in_mm, sensor_height_in_mm, alpha_u, alpha_v, u_0, v_0)
# # print(camm.data.sensor_width, camm.data.sensor_height)
# print(cam.data.angle)
# assert(0)

model_identifier = os.path.split(args.obj)[1]

# model_identifier = os.path.split(os.path.split(args.obj)[0])[1]
# model_file_identifier = os.path.split(os.path.split(os.path.split(args.obj)[0])[0])[1]

fp = os.path.join(args.output_folder, model_identifier.split('.')[0])

scene.render.image_settings.file_format = 'PNG'  # set output format to .png

stepsize = 360.0 / args.views
rotation_mode = 'XYZ'

for output_node in [depth_file_output]:#, normal_file_output, albedo_file_output]:
    output_node.base_path = ''

scene.render.use_multiview = True

for i in range(args.views):
    scene.frame_set(i)

    # angle_z = random.random() * 360
    # print(angle_z)
    # elu = mathutils.Euler((0.0, 0, math.radians(angle_z)), 'XYZ')
    # mat_rot = elu.to_matrix()
    # print(mat_rot)

    # R = mathutils.Matrix()
    # R[0][0:3] = mat_rot[0]
    # R[0][3] = R[0][2]
    # R[1][0:3] = mat_rot[1]
    # R[1][3] = R[1][2]
    # R[2][0:3] = mat_rot[2]
    # R[2][3] = R[2][2]

    # cam.matrix_world = R
    b_empty.rotation_euler[0] = math.radians(0 * random.random())
    b_empty.rotation_euler[2] = math.radians(360 * random.random())
    b_empty.rotation_euler[1] = math.radians(0 * random.random())

    scene.render.filepath = fp + '/' + str(i)

    depth_file_output.file_slots[0].path = scene.render.filepath + '_'
    # normal_file_output.file_slots[0].path = scene.render.filepath + "_normal.png"
    # albedo_file_output.file_slots[0].path = scene.render.filepath + "_albedo.png"
    bpy.ops.render.render(write_still=True)  # render still

    cam.rotation_euler = cam.matrix_world.to_euler('XYZ')[0:3]
    with open(scene.render.filepath +'.txt', 'w') as f:
        for i in range(4):
          for j in range(4):
            f.write(str(cam.matrix_world[i][j]))
            f.write('\n')
    f.close()

# BKE_camera_sensor_size
def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
      return sensor_y
    return sensor_x

# BKE_camera_sensor_fit
def get_sensor_fit(sensor_fit, size_x, size_y):
  if sensor_fit == 'AUTO':
    if size_x >= size_y:
      return 'HORIZONTAL'
    else:
      return 'VERTICAL'
  return sensor_fit


def get_calibration_matrix_K_from_blender(camd):
    scene = bpy.context.scene
    f_in_mm = camd.data.lens
    scale = scene.render.resolution_percentage / 100
    resolution_x_in_px = scale * scene.render.resolution_x
    resolution_y_in_px = scale * scene.render.resolution_y
    sensor_size_in_mm = get_sensor_size(camd.data.sensor_fit, camd.data.sensor_width, camd.data.sensor_height)
    sensor_fit = get_sensor_fit(
      camd.data.sensor_fit,
      scene.render.pixel_aspect_x * resolution_x_in_px,
      scene.render.pixel_aspect_y * resolution_y_in_px
    )
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
      view_fac_in_px = resolution_x_in_px
    else:
      view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
    s_u = 1 / pixel_size_mm_per_px
    s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

    # Parameters of intrinsic calibration matrix K
    u_0 = resolution_x_in_px / 2 - camd.data.shift_x * view_fac_in_px
    v_0 = resolution_y_in_px / 2 + camd.data.shift_y * view_fac_in_px / pixel_aspect_ratio
    skew = 0 # only use rectangular pixels

    K = np.array(
      ((s_u, skew, u_0),
      ( 0, s_v, v_0),
      ( 0, 0, 1)))
    return K

print("intrinsic_matrix:")
K = get_calibration_matrix_K_from_blender(cam)
print(K)