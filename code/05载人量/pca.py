import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from adjustText import adjust_text
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def create_pca_biplot(df):
    """
    创建PCA双标图，特别标记载人量和相关变量
    """
    # 选择数值型变量进行PCA分析
    numeric_columns = [
        'road_length_km', 'car_count', 'ebike_count', 'bus_count', 
        'pedestrian_count', 'avg_speed_kph', 'traffic_flow_vph', 
        'traffic_density_vpkm', 'congestion_index', 'passenger_throughput'
    ]
    
    # 提取数值数据
    X = df[numeric_columns].dropna()
    
    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 执行PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制观测点
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=50, color='lightblue')
    
    # 获取主成分载荷
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    # 绘制变量向量
    texts = []
    text_positions = []  # 记录文本位置避免重叠
    for i, var in enumerate(numeric_columns):
        # 配色方案调整
        if var == 'passenger_throughput':
            color = 'red'
            linewidth = 3
            alpha = 0.9
        elif var in ['traffic_flow_vph', 'traffic_density_vpkm', 'congestion_index', 'avg_speed_kph']:
            color = 'gold'
            linewidth = 2.5
            alpha = 0.8
        elif var == 'ebike_count':
            color = 'royalblue'
            linewidth = 2
            alpha = 0.7
        elif var in ['car_count', 'bus_count', 'pedestrian_count']:
            color = 'forestgreen'
            linewidth = 2
            alpha = 0.7
        elif var == 'road_length_km':
            color = 'gray'
            linewidth = 1.5
            alpha = 0.6
        else:
            color = 'gray'
            linewidth = 1.5
            alpha = 0.6

        # 绘制箭头
        ax.arrow(0, 0, loadings[i, 0], loadings[i, 1], 
                head_width=0.03, head_length=0.03, 
                fc=color, ec=color, linewidth=linewidth, alpha=alpha)
        
        # 中文变量名映射
        var_names = {
            'road_length_km': '道路长度',
            'car_count': '汽车数量',
            'ebike_count': '电动车数量',
            'bus_count': '公交车数量',
            'pedestrian_count': '步行者数量',
            'avg_speed_kph': '平均速度',
            'traffic_flow_vph': '交通流量',
            'traffic_density_vpkm': '交通密度',
            'congestion_index': '拥堵指数',
            'passenger_throughput': '载人量'
        }

        # 计算文本位置，避免重叠
        base_x = loadings[i, 0] * 1.2
        base_y = loadings[i, 1] * 1.2
        text_x, text_y = base_x, base_y
        for pos_x, pos_y in text_positions:
            distance = np.sqrt((text_x - pos_x)**2 + (text_y - pos_y)**2)
            if distance < 0.3:
                offset_x = 0.15 if base_x > 0 else -0.15
                offset_y = 0.15 if base_y > 0 else -0.15
                text_x = base_x + offset_x
                text_y = base_y + offset_y
        text_positions.append((text_x, text_y))

        fontweight = 'bold' if var == 'passenger_throughput' else 'normal'
        bbox_props = dict(
            boxstyle="round,pad=0.3",
            facecolor='white',
            edgecolor=color,
            linewidth=1.5,
            alpha=0.9
        )
        text = ax.text(text_x, text_y, var_names[var],
                       fontsize=11, color=color, fontweight=fontweight,
                       ha='center', va='center',
                       bbox=bbox_props)
        texts.append(text)
    
    # 设置坐标轴标签
    ax.set_xlabel(f'第一主成分 (解释方差: {pca.explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'第二主成分 (解释方差: {pca.explained_variance_ratio_[1]:.2%})')
    
    # 添加标题
    ax.set_title(f'载人量影响因素PCA双标图\n累计解释方差: {sum(pca.explained_variance_ratio_):.2%}', 
                fontsize=14, fontweight='bold')
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 添加原点线
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=3, label='载人量'),
        Line2D([0], [0], color='gold', lw=2.5, label='其他交通指标'),
        Line2D([0], [0], color='forestgreen', lw=2, label='其他车辆类型'),
        Line2D([0], [0], color='royalblue', lw=2, label='电动车'),
        Line2D([0], [0], color='gray', lw=1.5, label='其他道路参数')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return fig, ax, pca

def analyze_pca_results(pca, numeric_columns):
    """
    分析PCA结果，重点关注载人量
    """
    print("=== 载人量PCA分析结果 ===")
    print(f"各主成分解释方差比例:")
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {ratio:.3f} ({ratio*100:.1f}%)")
    
    print(f"\n累计解释方差: {sum(pca.explained_variance_ratio_):.3f} ({sum(pca.explained_variance_ratio_)*100:.1f}%)")
    
    # 变量载荷分析
    print("\n=== 载人量影响因素载荷分析 ===")
    loadings = pca.components_.T
    loadings_df = pd.DataFrame(loadings, 
                              columns=['PC1', 'PC2'], 
                              index=numeric_columns)
    
    print("PC1上载荷最大的变量:")
    pc1_sorted = loadings_df['PC1'].abs().sort_values(ascending=False)
    for var, loading in pc1_sorted.head(5).items():
        print(f"  {var}: {loadings_df.loc[var, 'PC1']:.3f}")
    
    print("\nPC2上载荷最大的变量:")
    pc2_sorted = loadings_df['PC2'].abs().sort_values(ascending=False)
    for var, loading in pc2_sorted.head(5).items():
        print(f"  {var}: {loadings_df.loc[var, 'PC2']:.3f}")
    
    # 特别分析载人量的载荷
    print(f"\n=== 载人量在主成分中的表现 ===")
    print(f"载人量在PC1上的载荷: {loadings_df.loc['passenger_throughput', 'PC1']:.3f}")
    print(f"载人量在PC2上的载荷: {loadings_df.loc['passenger_throughput', 'PC2']:.3f}")
    
    # 找出与载人量载荷方向最相似的变量
    passenger_loading = loadings_df.loc['passenger_throughput'].values
    correlations = {}
    for var in numeric_columns:
        if var != 'passenger_throughput':
            var_loading = loadings_df.loc[var].values
            correlation = np.dot(passenger_loading, var_loading) / (np.linalg.norm(passenger_loading) * np.linalg.norm(var_loading))
            correlations[var] = correlation
    
    print(f"\n与载人量载荷方向最相似的变量:")
    sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for var, corr in sorted_correlations[:5]:
        print(f"  {var}: {corr:.3f}")
    
    return loadings_df

# 示例使用代码
if __name__ == "__main__":
    # 读取数据文件
    csv_path = r"E:\github_projects\math_modeling\data\05载人量.csv"
    df = pd.read_csv(csv_path, encoding='utf-8')  # 如有乱码可尝试 encoding='gbk'

    # 如有必要，重命名或筛选列以适配 numeric_columns
    # 例如：df.rename(columns={'原列名':'road_length_km', ...}, inplace=True)
    # 或者只保留需要的列

    # 创建PCA双标图
    fig, ax, pca = create_pca_biplot(df)

    # 分析结果
    numeric_columns = [
        'road_length_km', 'car_count', 'ebike_count', 'bus_count', 
        'pedestrian_count', 'avg_speed_kph', 'traffic_flow_vph', 
        'traffic_density_vpkm', 'congestion_index', 'passenger_throughput'
    ]
    loadings_df = analyze_pca_results(pca, numeric_columns)

    # 显示图形
    plt.show()

    # 保存图片
    # fig.savefig('passenger_throughput_pca_biplot.png', dpi=300, bbox_inches='tight')
