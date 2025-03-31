import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os


def analyze_variance(all_stats, filenames):
    all_stats = np.array(all_stats)
    filenames = filenames
    print("\n==== all_stats 数据结构 ====")
    print("数据类型:", type(all_stats))
    print("数组维度:", all_stats.shape)
    print("示例数据 (前3个样本):\n", all_stats[:3])
    """执行方差分析和PCA"""

    # 方差分析
    variances = np.var(all_stats, axis=0)
    variances_plus = variances * 10
    plt.plot(variances_plus)
    plt.title('Variance per Degree')
    plt.xlabel('Spherical Harmonic Degree')
    plt.show()

    # 2. 控制台打印统计摘要
    print("\n==== 方差分析结果 ====")
    print(f"分析样本数: {all_stats.shape[0]}")
    print(f"最高方差: {np.max(variances):.4f} (degree {np.argmax(variances)})")
    print(f"最低方差: {np.min(variances):.4f} (degree {np.argmin(variances)})")
    print(f"平均方差: {np.mean(variances):.4f}")
    print(f"方差标准差: {np.std(variances):.4f}")

    # 3. 打印方差排名（前5和后5）
    sorted_indices = np.argsort(variances)[::-1]
    print("\n方差排名 Top 5:")
    for i, idx in enumerate(sorted_indices[:5]):
        print(f"Rank {i + 1}: degree {idx} -> {variances[idx]:.4f}")

    # 4. 输出完整度数方差表
    print("\n完整方差表：")
    print("Degree | Variance   | Percentage")
    print("-------------------------------")
    total_variance = np.sum(variances)
    for deg, var in enumerate(variances):
        print(f"{deg:5d} | {var:.6f} | {var / total_variance * 100:6.2f}%")


def analyze_pca(all_stats, filenames):
    # PCA分析

    # PCA分析（获取前4个主成分）
    pca = PCA(n_components=4)
    pca_result = pca.fit_transform(all_stats)

    # ==== 新增保存全部PCA坐标 ====
    if filenames is not None:
        pca_table = np.column_stack((
            filenames,
            pca_result[:, 0],  # PC1
            pca_result[:, 1],  # PC2
            pca_result[:, 2],  # PC3
            pca_result[:, 3]  # PC4
        ))
        #np.savetxt("pca_all_components.csv", pca_table,
                   #fmt="%s", delimiter=",",
                   #header="Filename,PC1,PC2,PC3,PC4")

    print("\n==== PCA载荷矩阵 ====")
    print("Shape:", pca.components_.shape)  # 应为(2,21)
    print("PC1载荷（按重要性排序）:")
    sorted_pc1 = np.argsort(np.abs(pca.components_[0]))[::-1]
    for deg in sorted_pc1:
        print(f"degree {deg}: {pca.components_[0, deg]:.3f}")

    print("\nPC2载荷（按重要性排序）:")
    sorted_pc2 = np.argsort(np.abs(pca.components_[1]))[::-1]
    for deg in sorted_pc2:
        print(f"degree {deg}: {pca.components_[1, deg]:.3f}")

    print("\nPC3载荷（按重要性排序）:")
    sorted_pc3 = np.argsort(np.abs(pca.components_[2]))[::-1]
    for deg in sorted_pc3:
        print(f"degree {deg}: {pca.components_[2, deg]:.3f}")

    print("\nPC4载荷（按重要性排序）:")
    sorted_pc4 = np.argsort(np.abs(pca.components_[3]))[::-1]
    for deg in sorted_pc4:
        print(f"degree {deg}: {pca.components_[3, deg]:.3f}")

    # 在analyze_features中调用
    plot_pca_components(pca_result, pca, filenames, x_component=0, y_component=1)  # PC1-PC2
    plot_pca_components(pca_result, pca, filenames, x_component=2, y_component=3)  # PC3-PC4


def plot_pca_components(pca_result, pca, filenames=None, x_component=0, y_component=1):
    """通用PCA绘图函数"""
    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(
        pca_result[:, x_component],
        pca_result[:, y_component],
        alpha=0.7,
        edgecolors='w',  # 白色边缘增强对比
        s=40            # 控制点大小
    )

    # 标注部分文件名
    if filenames is not None:
        for i, (x, y) in enumerate(zip(pca_result[:, x_component],
                                       pca_result[:, y_component])):
            plt.text(
                x, y,
                os.path.splitext(filenames[i])[0],
                fontsize=7,  # 更小字体
                alpha=0.8,  # 更高透明度
                rotation=20,  # 倾斜角度
                ha='left', va='center'
            )

    # 添加方差解释率
    explained = pca.explained_variance_ratio_
    plt.xlabel(f'PC{x_component + 1} ({explained[x_component]:.1%})')
    plt.ylabel(f'PC{y_component + 1} ({explained[y_component]:.1%})')

    plt.title(f'PCA Components {x_component + 1}-{y_component + 1}')
    plt.show()