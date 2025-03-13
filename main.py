import flowshape as fs
import igl
import numpy as np
import meshplot as mp
from igl import massmatrix, MASSMATRIX_TYPE_BARYCENTRIC, decimate, remove_unreferenced
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
from vedo import Plotter, Mesh, settings
import k3d
import pyvista as pv

"""
Before mapping
load the file and repair traiangle mesh
"""
# 加载并清理网格-read the file of triangle mesh
v, f = igl.read_triangle_mesh("E:/python_spharm/flowshape/code/demo_data/QSY_A_2797.stl")
v, f, _, _ = remove_unreferenced(v, f)

# repair triangle mesh
# 修复面索引（如果从1开始）
if f.min() == 1:
    f = f - 1

# 使用 trimesh 清理网格
mesh = trimesh.Trimesh(vertices=v, faces=f)
mesh.remove_infinite_values()

# 获取非退化面掩码（兼容新旧版本）
mask = mesh.nondegenerate_faces()
mask = mask.astype(bool)
mesh.update_faces(mask)

v, f = mesh.vertices, mesh.faces

# reduce target face on traiangle mesh
# 目标面数
target_faces = 20000
result = decimate(v, f, target_faces)

# 归一化前检查顶点-normalize
v, f = result[1], result[2]
v = fs.normalize(v)

"""
spherical parameterization
comformal mapping is applied
map the vertices to the sphere-calculate curvatures-
"""
# 运行球形映射- construct a sphere
sv = fs.sphere_map(v, f)

# 绘制并保存球形映射图像-spherical mapping plot
plotter = mp.Viewer(settings={"background": "#FFFFFF", "antialias": True})
plotter.add_mesh(sv, f, shading={"wireframe": True})
plotter.save("sphere_map.html")

# 计算曲率-calculating curvature
rho = fs.curvature_function(v, sv, f)

# 绘制并保存曲率图像-plot based on curvature
plotter = mp.Viewer(settings={"background": "#FFFFFF", "antialias": True})  # 白色背景 + 抗锯齿
plotter.add_mesh(v, f, c=rho, shading={"wireframe": True})
plotter.save("curvature_plot.html")

# 根据曲率重建形状-reconstrct shape based on curvature
rv = fs.reconstruct_shape(sv, f, rho)

# 绘制并保存重建形状图像
plotter = mp.Viewer(settings={"background": "#FFFFFF", "antialias": True})
plotter.add_mesh(rv, f, c=rho, shading={"wireframe": True})
plotter.save("reconstructed_shape.html")

"""
spherical harmonics decomposation
"""
# 计算顶点球谐函数-calculate SPHARM on vertices
weights, Y_mat, vs = fs.do_mapping(v, f, l_max=40)

# 生成 degree 和 order 信息-generate a csv file to save l/m/weight
degree = []
order = []
for l in range(40):  # l 从 0 到 34（共35个值）
    for m in range(-l, l + 1):  # m 从 -l 到 l
        degree.append(l)
        order.append(m)
weights_df = pd.DataFrame({
    'degree': degree,
    'order': order,
    'weight': weights})
weights_df.to_csv("weights_with_degree_and_order.csv", index=False)
print("weights_with_degree_and_order.csv has been saved")

"""
reconstruction from SPHARM
"""
# 使用球形基函数的矩阵计算 rho2- new curvature according to SPHARM
rho2 = Y_mat.dot(weights)

# 绘制并保存基于新 rho2 的映射图像- new curvature mapping
plotter_rho2 = mp.Viewer(settings={"background": "#FFFFFF", "antialias": True})
plotter_rho2.add_mesh(sv, f, c=rho2, shading={"wireframe": True})
plotter_rho2.save("mapped_rho2.html")

# 重建新的形状- reconstruct shape based on new curvature-meshplot-viewer
# rec2 = fs.reconstruct_shape(sv, f, rho2)
# plotter_rec2 = mp.Viewer(settings={"background": "#FFFFFF", "antialias": True})
# plotter_rec2.add_mesh(rec2, f, c=rho2, shading={"wireframe": True})
# plotter_rec2.save("reconstructed_shape2.html")

# by vedo (Plotter)
# curvature_mesh = Mesh([v, f])
# curvature_mesh.celldata["curvature"] = rho2 # 赋值曲率数据
# curvature_mesh.cmap("coolwarm", "curvature", on='cells') # 颜色映射
# plotter = Plotter(bg="white") # 创建绘图对象
# plotter.show(curvature_mesh, "Face-based Curvature", interactive=True)

# by pyvista
# 生成面索引数组
faces_pv = np.insert(f, 0, 3, axis=1).astype(int)
faces_pv = faces_pv.flatten()
# 创建 PyVista 网格
mesh_pv = pv.PolyData(v, faces=faces_pv)
# 绑定面曲率数据
mesh_pv.cell_data["curvature2"] = rho2
# 映射颜色到面
plotter = pv.Plotter()
plotter.add_mesh(
    mesh_pv,
    scalars="curvature2",
    cmap="coolwarm",
    clim=(np.min(rho2), np.max(rho2)),
    show_edges=False,
)
plotter.show()
