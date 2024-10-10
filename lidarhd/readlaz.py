# sudo pacman -Sy laszip
# pip install laspy laszip --break-system-packages


import laspy
import open3d as o3d
import numpy as np

las = laspy.read("LHD_FXX_0985_6467_PTS_O_LAMB93_IGN69.copc.laz")
las

las.header
las.header.point_format
las.header.point_count
las.vlrs

list(las.point_format.dimension_names)

# filter out only the building class
# Classe 1 : Non classé
# Classe 2 : Sol
# Classe 3 : Végétation basse
# Classe 4 : Végétation moyenne
# Classe 5 : Végétation haute
# Classe 6 : Bâtiment
# Classe 9 : Eau
# Classe 17 : Pont
# Classe 64 : Sursol pérenne
# Classe 65 : Artefacts
# Classe 66 : Points virtuels
buildings = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
buildings.points = las.points[las.classification == 6]


print(np.min(las.X), np.max(las.X))
print(np.min(las.Y), np.max(las.Y))
region = np.where( las.X<(np.min(las.X)+(np.max(las.X)-np.min(las.X))/10)) and np.where(las.Y<(np.min(las.Y)+(np.max(las.Y)-np.min(las.Y))/10))
point_data = np.stack([ [las.X[k] for k in region][0], [las.Y[k] for k in region][0], [las.Z[k] for k in region][0] ], axis=0).transpose((1, 0))

point_data = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))
# point_data = np.stack([las.X, las.Y, las.classification], axis=0).transpose((1, 0))

geom = o3d.geometry.PointCloud()
geom.points = o3d.utility.Vector3dVector(point_data)
o3d.visualization.draw_geometries([geom])

import matplotlib.pyplot as plt
plt.plot(las.X)
plt.show()
