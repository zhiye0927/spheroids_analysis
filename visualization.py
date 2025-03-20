import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt


def visualize_power_spectrum(stat_df, max_degree=None, log_scale=True, filename=None):
    """

    参数：
    stat_df : DataFrame
        来自compute_degree_spectrum的统计表格
    max_degree : int 或 None, 可选
        显示的最大阶数（默认显示所有阶数）
    log_scale : bool, 可选
        是否使用对数坐标轴（默认True）
    filename : str, 可选
        图片保存路径（默认不保存）
    """
    # 数据准备
    df = stat_df.copy()
    if max_degree is not None:
        df = df[df['degree'] <= max_degree]

    # 创建图形
    plt.figure(figsize=(12, 6))

    # 绘制功率曲线
    plt.plot(df['degree'], df['total_power'],
             marker='o',
             linestyle='-',
             color='#2c7bb6',
             markersize=6,
             linewidth=2,
             label='Total Power')

    # 绘制最大振幅曲线
    plt.plot(df['degree'], df['max_amplitude'],
             marker='s',
             linestyle='--',
             color='#d7191c',
             markersize=5,
             linewidth=1.5,
             alpha=0.7,
             label='Max Amplitude')

    # 设置坐标轴
    plt.xlabel('Spherical Harmonic Degree (l)', fontsize=12, labelpad=10)
    plt.ylabel('Power / Amplitude' + (' (log scale)' if log_scale else ''), fontsize=12, labelpad=10)
    plt.title('Spherical Harmonic Energy Spectrum', fontsize=14, pad=20)

    # 设置刻度
    plt.xticks(np.arange(0, df['degree'].max() + 1, 5 if df['degree'].max() > 20 else 2))
    plt.xlim(-0.5, df['degree'].max() + 0.5)

    # 对数坐标处理
    if log_scale:
        plt.yscale('log')
        plt.grid(True, which="both", ls="--", alpha=0.3)
    else:
        plt.grid(True, axis='y', ls="--", alpha=0.3)

    # 添加图例
    plt.legend(fontsize=10, frameon=True, loc='upper right')

    # 保存或显示
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形
        print(f"频谱图已保存至：{filename}")
    else:
        plt.show()


# Function to reconstruct shape according to spherical harmonics
def visualize_spherical_harmonics_reconstruction(grid_sh):
    """使用球谐函数重建的3D形状可视化"""
    # 获取半径数据
    grid_data = np.real(grid_sh.data)
    grid_size = grid_data.shape[0]

    # 生成球面坐标
    theta = np.linspace(0, np.pi, grid_size, endpoint=True)  # 0 到 π
    phi = np.linspace(0, 2 * np.pi, grid_size, endpoint=True)  # 0 到 2π
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')

    # 计算笛卡尔坐标
    r = grid_data  # 重建的半径
    x = (r * np.sin(theta_grid) * np.cos(phi_grid)).T
    y = (r * np.sin(theta_grid) * np.sin(phi_grid)).T
    z = (r * np.cos(theta_grid)).T

    # 创建 Pyvista 网格
    grid = pv.StructuredGrid(x, y, z)

    # 绘制 3D 形状
    plotter = pv.Plotter()
    plotter.add_mesh(grid,
                     scalars=r.flatten(),
                     cmap="coolwarm",
                     opacity=1.0,
                     show_edges=True,
                     specular=0.8)
    plotter.add_axes(box_args={'color': 'red'})
    plotter.add_title(f"球谐重建形状\n分辨率: {grid_size}x{grid_size}")
    plotter.show()
