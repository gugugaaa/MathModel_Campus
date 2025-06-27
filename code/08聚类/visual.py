import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_clusters(X, labels, centers=None, title="聚类结果可视化"):
    """
    可视化聚类结果
    :param X: 数据点，形状为 (n_samples, 2)
    :param labels: 聚类标签，形状为 (n_samples,)
    :param centers: 聚类中心，形状为 (n_clusters, 2)，可选
    :param title: 图标题
    """
    X = np.asarray(X)
    labels = np.asarray(labels)
    n_clusters = len(np.unique(labels))
    plt.figure(figsize=(8, 6))
    colors = plt.cm.get_cmap('tab10', n_clusters)
    for i in range(n_clusters):
        plt.scatter(X[labels == i, 0], X[labels == i, 1], 
                    s=30, color=colors(i), label=f'簇 {i}')
    if centers is not None:
        centers = np.asarray(centers)
        plt.scatter(centers[:, 0], centers[:, 1], 
                    s=200, c='black', marker='X', label='聚类中心')
    plt.title(title)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.grid(True)
    plt.show()

def load_clustered_data():
    """
    加载已分好类的三个表格，返回三个DataFrame
    :return: (df_low, df_mid, df_high)
    """
    df_low = pd.read_csv(r'data\08聚类\01低拥堵.csv')
    df_mid = pd.read_csv(r'data\08聚类\02中拥堵.csv')
    df_high = pd.read_csv(r'data\08聚类\03高拥堵.csv')
    return df_low, df_mid, df_high

def main():
    # 设置matplotlib中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
    plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

    # 加载数据
    df_low, df_mid, df_high = load_clustered_data()
    # 假设每个表格有两列用于聚类可视化，例如 'x', 'y'
    X_low = df_low.iloc[:, :2].values
    X_mid = df_mid.iloc[:, :2].values
    X_high = df_high.iloc[:, :2].values

    # 合并数据和标签
    X = np.vstack([X_low, X_mid, X_high])
    labels = np.concatenate([
        np.full(len(X_low), 0),
        np.full(len(X_mid), 1),
        np.full(len(X_high), 2)
    ])

    plot_clusters(X, labels, title="聚类结果可视化")

if __name__ == "__main__":
    main()
