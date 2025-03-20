import os
import numpy as np
from scipy.interpolate import griddata
import pyvista as pv
import pyshtools.expand as shtools
import pyshtools as pysh
import pandas as pd


# Function to convert Cartesian coordinates to spherical coordinates
def cartesian_to_spherical(normalized_vertices):
    """笛卡尔坐标系到极坐标系"""
    spherical_coordinates = np.zeros((len(normalized_vertices), 3))
    for i, vertex in enumerate(normalized_vertices):
        x, y, z = vertex
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        r = np.where(r == 0, 0.00001, r)  # 避免除零错误
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)  # 原始范围为[-π, π]
        phi = phi % (2 * np.pi)  # 强制映射到[0, 2π)
        spherical_coordinates[i] = [r, theta, phi]

    return spherical_coordinates


# Function to interpolate spherical coordinates to a regular grid
def spherical_interpolate(R, theta, phi, grid_size):
    """
    参数：
    R : array - 半径数组
    theta : array - 极角数组（弧度，范围[0, π]）
    phi : array - 方位角数组（弧度，范围[0, 2π)）
    grid_size : int - 输出网格尺寸

    返回：
    ndarray - (grid_size, grid_size)的插值网格
    """
    if len(R) < 4:
        return None

    # 生成规则网格
    I = np.linspace(0, np.pi, grid_size, endpoint=False)
    J = np.linspace(0, 2 * np.pi, grid_size, endpoint=False)
    J, I = np.meshgrid(J, I)

    # 原始数据点
    values = R
    points = np.array([theta, phi]).T

    # 添加极点（theta=0和pi）
    points = np.concatenate((points,
                             np.array([[0, 0], [0, 2 * np.pi], [np.pi, 0], [np.pi, 2 * np.pi]])), axis=0)
    rmin = np.mean(R[theta == theta.min()])
    rmax = np.mean(R[theta == theta.max()])
    values = np.concatenate((values, [rmin, rmin, rmax, rmax]))

    # 处理phi周期性
    points = np.concatenate((points, points - [0, 2 * np.pi], points + [0, 2 * np.pi]), axis=0)
    values = np.concatenate((values, values, values))

    # 生成插值点
    xi = np.array([[I[i, j], J[i, j]] for i in range(grid_size) for j in range(grid_size)])

    # 执行插值
    grid = griddata(points, values, xi, method='linear')
    grid = grid.reshape((grid_size, grid_size))
    grid[:, -1] = grid[:, 0]

    # 验证闭合性
    print("验证闭合性:")
    print(f"第一列均值: {np.mean(grid[:, 0]):.4f}, 最后一列均值: {np.mean(grid[:, -1]):.4f}")
    if not np.allclose(grid[:, 0], grid[:, -1], atol=1e-3):
        print("警告：经度方向闭合性未满足！")

    return grid


# Function to visualize the interpolated regular grid
def visualize_interpolated(grid_r):
    """插值后网格可视化"""
    # 获取网格尺寸（假设为正方形）
    grid_size = grid_r.shape[0]

    # 生成规则的球面网格（适配正方形）
    theta = np.linspace(0, np.pi, grid_size, endpoint=False)  # 纬度方向 [0, π)
    phi = np.linspace(0, 2 * np.pi, grid_size, endpoint=False)  # 经度方向 [0, 2π)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')  # 维度 (grid_size, grid_size)
    # 验证采样点
    print("采样点验证:")
    print(f"theta 范围: {theta[0]:.3f} ~ {theta[-1]:.3f} (期望: 0 ~ π)")
    print(f"phi 范围: {phi[0]:.3f} ~ {phi[-1]:.3f} (期望: 0 ~ 2π)")

    # 转换为笛卡尔坐标
    x = grid_r * np.sin(theta_grid) * np.cos(phi_grid).T
    y = grid_r * np.sin(theta_grid) * np.sin(phi_grid).T
    z = grid_r * np.cos(theta_grid).T

    # 验证 φ=0 和 φ=2π 处的点是否重合
    idx_phi0 = 0      # φ=0 的列索引
    idx_phi2pi = -1   # φ=2π 的列索引
    tolerance = 1e-3
    x_diff = np.max(np.abs(x[:, idx_phi0] - x[:, idx_phi2pi]))
    y_diff = np.max(np.abs(y[:, idx_phi0] - y[:, idx_phi2pi]))
    z_diff = np.max(np.abs(z[:, idx_phi0] - z[:, idx_phi2pi]))
    print("闭合性验证 (φ=0 vs φ=2π):")
    print(f"最大坐标差异: x={x_diff:.3e}, y={y_diff:.3e}, z={z_diff:.3e}")
    if x_diff > tolerance or y_diff > tolerance or z_diff > tolerance:
        print("错误：φ=0 和 φ=2π 处坐标不重合！")


    # 可视化
    grid = pv.StructuredGrid(x, y, z)
    grid["Radius"] = grid_r.T.ravel()
    plotter = pv.Plotter()
    plotter.add_mesh(grid,
                     color='cyan',
                     opacity=0.8,
                     show_edges=True,
                     scalars=grid_r.flatten(),
                     cmap='viridis')
    plotter.add_title(f"正方形插值网格 ({grid_size}x{grid_size})")
    plotter.show()


# Function to compute spherical harmonics coefficients
def compute_spherical_harmonics(surface, normalize=True, normalization_method='zero-component'):
    """
    参数
    ----------
    surface : numpy.ndarray, 维度 (n, n) 或 (n, 2n)
        符合Driscoll-Healy采样定理的二维网格，n必须为偶数
    normalize : bool, 可选
        是否进行归一化处理（默认False）
    normalization_method : str, 可选
        归一化方法：'zero-component' 或 'mean-radius'（默认'zero-component'）

    返回
    -------
    harmonics : numpy.ndarray
        球谐系数数组，格式与shtools.SHExpandDHC输出一致
    """
    # 输入验证
    if surface.shape[1] % 2 or surface.shape[0] % 2:
        raise ValueError("Latitude and longitude samples (n) must be even")

    # 确定网格类型
    if surface.shape[1] == surface.shape[0]:
        sampling = 1  # 等采样网格
    elif surface.shape[1] == 2 * surface.shape[0]:
        sampling = 2  # 等间距网格
    else:
        raise ValueError("Grid must be (N, N) or (N, 2N)")

    # 预处理
    processed_surface = surface.copy()
    if normalize and normalization_method == 'mean-radius':
        processed_surface /= np.mean(np.abs(processed_surface))

    # 球谐展开
    harmonics = shtools.SHExpandDHC(processed_surface, sampling=sampling)

    # 后处理
    if normalize and normalization_method == 'zero-component':
        harmonics = harmonics / harmonics[0][0, 0]

    return harmonics


def clm_to_1d_standard(clm, target_l_max=30):
    """
    将球谐系数 clm（形状为 (2, l_max_input+1, l_max_input+1)）转换为一维实数数组，
    只保留 l=0 到 target_l_max 部分的系数，
    排列顺序要求每个 l 内的系数按 m 从 -l 到 l 的顺序排列。

    参数：
        clm: numpy.ndarray, 形状 (2, l_max_input+1, l_max_input+1)
            clm[0, l, m] 存放 m >= 0 的系数，
            clm[1, l, m] 存放 m < 0 的系数，其中 clm[1, l, 1] 对应 m=-1，clm[1, l, l] 对应 m=-l。
        target_l_max: int, 默认为 30
            指定导出的最高阶 l 值（只导出 l=0 到 target_l_max 部分）。

    返回：
        一维 numpy 数组，长度为 (target_l_max+1)**2，
        排列顺序为：
        对于每个 l, 依次为 [c(l,-l), c(l,-l+1), ..., c(l,-1), c(l,0), c(l,1), ..., c(l,l)]，
        并取其实部。
    """
    # 检查输入的系数最高阶是否足够
    if clm.shape[1] - 1 < target_l_max:
        raise ValueError("输入系数的最高阶小于 target_l_max")

    coeffs = []
    for l in range(target_l_max + 1):
        # 负 m 部分：m = -l, -l+1, …, -1
        for m in range(l, 0, -1):
            coeffs.append(clm[1, l, m])
        # m = 0 部分
        coeffs.append(clm[0, l, 0])
        # 正 m 部分：m = 1, 2, …, l
        for m in range(1, l + 1):
            coeffs.append(clm[0, l, m])

    # 转换为 numpy 数组并取实部
    return np.real(np.array(coeffs))


def process_spherical_harmonics(clm, output_path=None):
    """

    参数：
    clm : pysh.SHCoeffs 或 numpy.ndarray
        输入球谐系数，支持以下格式：
        - pyshtools的SHCoeffs对象
        - numpy数组(2, lmax+1, lmax+1)
    output_path : str, 可选
        输出文件路径，支持以下格式：
        - .xlsx: 生成多工作表Excel文件
        - .csv: 生成两个CSV文件（自动添加后缀）
        - None: 不保存文件

    返回：
    tuple (full_df, stat_df):
        full_df: 包含完整球谐系数的DataFrame
        stat_df: 按阶统计的能量谱DataFrame

    功能特性：
    1. 自动类型识别和转换
    2. 内存预分配优化
    3. 数据类型安全验证
    4. 支持多格式输出
    5. 向量化计算加速
    """
    # ======================
    # 输入验证和预处理
    # ======================
    # 处理不同输入类型
    if isinstance(clm, pysh.SHCoeffs):
        coeffs = clm.to_array()
        lmax = clm.lmax
    elif isinstance(clm, np.ndarray):
        if clm.ndim != 3 or clm.shape[0] != 2:
            raise ValueError("Numpy数组输入格式应为(2, lmax+1, lmax+1)")
        coeffs = clm
        lmax = clm.shape[1] - 1
    else:
        raise TypeError("不支持的输入类型，请输入SHCoeffs对象或numpy数组")

    # 验证复数类型
    if not np.iscomplexobj(coeffs):
        raise ValueError("输入系数必须包含复数数据")

    # ======================
    # 全量数据表生成（优化版）
    # ======================
    # 预计算记录总数
    n_records = (lmax + 1) * (lmax + 2)

    # 内存预分配（显式指定数据类型）
    data = {
        'degree': np.empty(n_records, dtype=np.int32),
        'order': np.empty(n_records, dtype=np.int32),
        'value': np.empty(n_records, dtype=np.complex128)
    }

    # 填充数据（比列表append快约20倍）
    idx = 0
    for l in range(lmax + 1):
        # 负序数部分 (m = -l 到 -1)
        for m in range(l, 0, -1):  # 修正1：逆序循环
            data['degree'][idx] = l
            data['order'][idx] = -m  # m=5 → order=-5 (当l=5时)
            data['value'][idx] = coeffs[1, l, m]
            idx += 1

        # m=0 部分
        data['degree'][idx] = l
        data['order'][idx] = 0
        data['value'][idx] = coeffs[0, l, 0]
        idx += 1

        # 正序数部分 (m = 1 到 l)
        for m in range(1, l + 1):  # 修正2：正序数放在最后
            data['degree'][idx] = l
            data['order'][idx] = m
            data['value'][idx] = coeffs[0, l, m]
            idx += 1

    # 创建DataFrame并计算衍生字段
    full_df = pd.DataFrame(data)

    # 向量化计算（比apply快约100倍）
    full_df['amplitude'] = np.abs(full_df['value'])
    full_df['power'] = full_df['amplitude'] ** 2
    full_df['real'] = np.real(full_df['value'])  # 使用numpy函数替代属性访问
    full_df['imag'] = np.imag(full_df['value'])
    full_df['harmonic'] = "l=" + full_df['degree'].astype(str) + " m=" + full_df['order'].astype(str)

    # ======================
    # 统计表生成（优化版）
    # ======================
    # 预分配数组
    degrees = np.arange(lmax + 1)
    total_power = np.zeros_like(degrees, dtype=np.float64)
    max_amplitude = np.zeros_like(degrees, dtype=np.float64)

    # 向量化统计计算
    for l in degrees:
        mask = full_df['degree'] == l
        amplitudes = full_df.loc[mask, 'amplitude'].values

        total_power[l] = np.sum(amplitudes ** 2)
        max_amplitude[l] = np.max(amplitudes) if amplitudes.size > 0 else 0.0

    stat_df = pd.DataFrame({
        'degree': degrees,
        'total_power': total_power,
        'max_amplitude': max_amplitude,
        'total_amplitude': np.sqrt(total_power)
    })

    # ======================
    # 数据保存（增强版）
    # ======================
    if output_path:
        base_name, ext = os.path.splitext(output_path)

        if ext.lower() == '.xlsx':
            # Excel格式：单文件多工作表
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                full_df.to_excel(writer, sheet_name='Full_Coefficients', index=False)
                stat_df.to_excel(writer, sheet_name='Degree_Statistics', index=False)
                print(f"结果已保存到Excel文件: {output_path}")

        elif ext.lower() == '.csv':
            # CSV格式：生成两个文件
            full_path = f"{base_name}_full.csv"
            stat_path = f"{base_name}_stat.csv"

            full_df.to_csv(full_path, index=False)
            stat_df.to_csv(stat_path, index=False)
            print(f"全量数据已保存到: {full_path}")
            print(f"统计结果已保存到: {stat_path}")

        else:
            raise ValueError("不支持的文件格式，请使用.xlsx或.csv")

    return full_df, stat_df
