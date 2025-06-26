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

def create_co2_pca_biplot(df):
    """
    创建CO2排放量PCA双标图，特别标记CO2排放量和相关影响因素
    """
    # 选择数值型变量进行PCA分析
    numeric_columns = [
        'road_length_km', 'car_count', 'ebike_count', 'bus_count', 
        'pedestrian_count', 'avg_speed_kph', 'traffic_flow_vph', 
        'traffic_density_vpkm', 'congestion_index', 'passenger_throughput', 'co2_emission'
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
    for i, var in enumerate(numeric_columns):
        # 为特殊变量设置颜色
        if var == 'co2_emission':
            color = 'red'
            linewidth = 3
            alpha = 0.9
        elif var in ['car_count', 'traffic_flow_vph', 'congestion_index']:
            color = 'orange'
            linewidth = 2.5
            alpha = 0.8
        elif var in ['avg_speed_kph', 'ebike_count']:
            color = 'green'
            linewidth = 2
            alpha = 0.7
        else:
            color = 'darkblue'
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
            'passenger_throughput': '载客量',
            'co2_emission': 'CO2排放量'
        }
        
        # 添加变量标签
        text = ax.text(loadings[i, 0]*1.1, loadings[i, 1]*1.1, 
                      var_names[var], fontsize=10, 
                      color=color, 
                      fontweight='bold' if var in ['co2_emission', 'car_count', 'traffic_flow_vph', 'congestion_index'] else 'normal')
        texts.append(text)
    
    # 调整文本位置避免重叠
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))
    
    # 设置坐标轴标签
    ax.set_xlabel(f'第一主成分 (解释方差: {pca.explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'第二主成分 (解释方差: {pca.explained_variance_ratio_[1]:.2%})')
    
    # 添加标题
    ax.set_title(f'CO2排放量交通数据PCA双标图\n累计解释方差: {sum(pca.explained_variance_ratio_):.2%}', 
                fontsize=14, fontweight='bold')
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 添加原点线
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=3, label='CO2排放量'),
        Line2D([0], [0], color='orange', lw=2.5, label='主要影响因素'),
        Line2D([0], [0], color='green', lw=2, label='速度/电动车'),
        Line2D([0], [0], color='darkblue', lw=1.5, label='其他变量')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return fig, ax, pca

def analyze_co2_pca_results(pca, numeric_columns):
    """
    分析CO2相关的PCA结果
    """
    print("=== CO2排放量PCA分析结果 ===")
    print(f"各主成分解释方差比例:")
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {ratio:.3f} ({ratio*100:.1f}%)")
    
    print(f"\n累计解释方差: {sum(pca.explained_variance_ratio_):.3f} ({sum(pca.explained_variance_ratio_)*100:.1f}%)")
    
    # 变量载荷分析
    print("\n=== 变量载荷分析 ===")
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
    
    # CO2排放量与其他变量的关系分析
    print("\n=== CO2排放量相关性分析 ===")
    co2_pc1 = loadings_df.loc['co2_emission', 'PC1']
    co2_pc2 = loadings_df.loc['co2_emission', 'PC2']
    print(f"CO2排放量在PC1上的载荷: {co2_pc1:.3f}")
    print(f"CO2排放量在PC2上的载荷: {co2_pc2:.3f}")
    
    # 寻找与CO2排放量方向最相似的变量
    print("\n与CO2排放量方向最相似的变量:")
    co2_vector = np.array([co2_pc1, co2_pc2])
    similarities = {}
    for var in numeric_columns:
        if var != 'co2_emission':
            var_vector = np.array([loadings_df.loc[var, 'PC1'], loadings_df.loc[var, 'PC2']])
            # 计算余弦相似度
            similarity = np.dot(co2_vector, var_vector) / (np.linalg.norm(co2_vector) * np.linalg.norm(var_vector))
            similarities[var] = similarity
    
    sorted_similarities = sorted(similarities.items(), key=lambda x: abs(x[1]), reverse=True)
    for var, sim in sorted_similarities[:5]:
        direction = "正相关" if sim > 0 else "负相关"
        print(f"  {var}: {sim:.3f} ({direction})")
    
    return loadings_df

# 示例使用代码
if __name__ == "__main__":
    # 读取你的CSV数据
    csv_path = r"E:\github_projects\math_modeling\data\06二氧化碳.csv"
    df = pd.read_csv(csv_path, encoding='utf-8')  # 如有乱码可尝试 encoding='gbk'

    # 如有必要，重命名或筛选列以适配 numeric_columns
    # 例如：df.rename(columns={'原列名':'road_length_km', ...}, inplace=True)

    # 创建CO2 PCA双标图
    fig, ax, pca = create_co2_pca_biplot(df)

    # 分析结果
    numeric_columns = [
        'road_length_km', 'car_count', 'ebike_count', 'bus_count', 
        'pedestrian_count', 'avg_speed_kph', 'traffic_flow_vph', 
        'traffic_density_vpkm', 'congestion_index', 'passenger_throughput', 'co2_emission'
    ]
    loadings_df = analyze_co2_pca_results(pca, numeric_columns)

    # 显示图形
    plt.show()

    # 保存图片
    # fig.savefig('co2_pca_biplot.png', dpi=300, bbox_inches='tight')
