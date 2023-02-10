import os
import random
from typing_extensions import Self
fir_path = '/home/albert/dataset/house3K'
firs = os.listdir(fir_path)
firs.sort()

# for i in range(1):
#     if firs[i].endswith('.obj'):
#         prob = random.random()
#         if prob > 0.2:
#             obj = os.path.join(fir_path, firs[i])
#             os.system('blender --background --python render_blender.py -- --views 10 %s --output_folder /home/albert/dataset/house3K_partial/train' % obj)
    
#         elif prob > 0.1 and prob < 0.2:
#             obj = os.path.join(fir_path, firs[i])
#             os.system('blender --background --python render_blender.py -- --views 10 %s --output_folder /home/albert/dataset/house3K_partial/valid' % obj)
        
#         else:
#             obj = os.path.join(fir_path, firs[i])
#             os.system('blender --background --python render_blender.py -- --views 10 %s --output_folder /home/albert/dataset/house3K_partial/test' % obj)

for i in range(5):
    if firs[i].endswith('.obj'):

        obj = os.path.join(fir_path, firs[i])
        os.system('blender --background --python render_blender.py -- --views 10 %s --output_folder /home/albert/dataset/house3K_partial/train' % obj)