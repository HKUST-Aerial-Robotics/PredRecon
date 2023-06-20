import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('X')
ax.set_xlim3d(-10, 10)
ax.set_ylabel('Y')
ax.set_ylim3d(-10, 10)
ax.set_zlabel('Z')
ax.set_zlim3d(-10, 10)

ax.quiver(0,0,0, 3.0,4.0,5.0,color=(1,0,0,0.5)) 
plt.show()