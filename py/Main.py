import matplotlib.pyplot as plt
import numpy as np

import MSA as MSA
import VJM as VJM


def plotDeflectionMap(x_pos, y_pos, z_pos, deflection, colormap, s):
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlim3d(0, 1)
    ax.set_ylim3d(0, 1)
    ax.set_zlim3d(0, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.colorbar(ax.scatter3D(x_pos, y_pos, z_pos, c=deflection, cmap=colormap, s=s))
    plt.show()


print("MSA result:")

x, y, z, d_msa, time = MSA.Main()
print(x[50], y[50], z[50], d_msa[50])
print("Exec time: {}".format(time))

print()

print("VJM result:")


x, y, z, d_vjm, time = VJM.Main()
print(x[50], y[50], z[50], d_vjm[50])
print("Exec time: {}".format(time))

deflection_diff = d_vjm - d_msa

print()
print("Difference")
clmap = plt.cm.get_cmap('viridis', 12)
plotDeflectionMap(x, y, z, deflection_diff, clmap, 60)

