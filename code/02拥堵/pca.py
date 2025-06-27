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
    创建PCA双标图，特别标记拥堵指数和电动车数量
    """
    # 选择数值型变量进行PCA分析
    numeric_columns = [
        'road_length_km', 'car_count', 'ebike_count', 'bus_count', 
        'pedestrian_count', 'avg_speed_kph', 'traffic_flow_vph', 
        'traffic_density_vpkm', 'congestion_index'
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
        # 重新设计配色方案 - 突出核心研究变量
        if var == 'congestion_index':
            color = '#FF4500'  # 橙红色 - 拥堵指数（核心目标变量）
            linewidth = 3.0
            alpha = 0.9
        elif var == 'ebike_count':
            color = '#1E90FF'  # 道奇蓝 - 电动车数量（核心研究变量）
            linewidth = 3.0
            alpha = 0.9
        elif var in ['avg_speed_kph', 'traffic_flow_vph', 'traffic_density_vpkm']:
            color = '#DC143C'  # 深红色 - 交通性能指标
            linewidth = 2.0
            alpha = 0.8
        elif var in ['car_count', 'bus_count']:
            color = '#8A2BE2'  # 蓝紫色 - 机动车辆
            linewidth = 2.0
            alpha = 0.7
        elif var == 'pedestrian_count':
            color = '#228B22'  # 森林绿 - 行人（非机动）
            linewidth = 2.0
            alpha = 0.7
        else:  # road_length_km
            color = '#696969'  # 暗灰色 - 基础设施
            linewidth = 1.8
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
            'congestion_index': '拥堵指数'
        }
        
        # 添加变量标签
        fontweight = 'bold' if var in ['congestion_index', 'ebike_count'] else 'normal'
        text = ax.text(loadings[i, 0]*1.1, loadings[i, 1]*1.1, 
                      var_names[var], fontsize=10, 
                      color=color, fontweight=fontweight)
        texts.append(text)
    
    # 调整文本位置避免重叠
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))
    
    # 设置坐标轴标签
    ax.set_xlabel(f'第一主成分 (解释方差: {pca.explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'第二主成分 (解释方差: {pca.explained_variance_ratio_[1]:.2%})')
    
    # 添加标题
    ax.set_title(f'交通数据PCA双标图\n累计解释方差: {sum(pca.explained_variance_ratio_):.2%}', 
                fontsize=14, fontweight='bold')
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 添加原点线
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#FF4500', lw=3, label='拥堵指数（目标变量）'),
        Line2D([0], [0], color='#1E90FF', lw=3, label='电动车数量（研究变量）'),
        Line2D([0], [0], color='#DC143C', lw=2, label='交通性能指标'),
        Line2D([0], [0], color='#8A2BE2', lw=2, label='机动车辆'),
        Line2D([0], [0], color='#228B22', lw=2, label='行人'),
        Line2D([0], [0], color='#696969', lw=1.8, label='基础设施')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    return fig, ax, pca

def analyze_pca_results(pca, numeric_columns):
    """
    分析PCA结果
    """
    print("=== PCA分析结果 ===")
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
    for var, loading in pc1_sorted.head(3).items():
        print(f"  {var}: {loadings_df.loc[var, 'PC1']:.3f}")
    
    print("\nPC2上载荷最大的变量:")
    pc2_sorted = loadings_df['PC2'].abs().sort_values(ascending=False)
    for var, loading in pc2_sorted.head(3).items():
        print(f"  {var}: {loadings_df.loc[var, 'PC2']:.3f}")
    
    return loadings_df

# 示例使用代码
if __name__ == "__main__":
    # 读取你的CSV数据
    csv_path = r"E:\github_projects\math_modeling\data\03拥挤.csv"
    df = pd.read_csv(csv_path, encoding='utf-8')  # 如有乱码可尝试 encoding='gbk'

    # 如有必要，重命名或筛选列以适配 numeric_columns
    # 例如：df.rename(columns={'原列名':'road_length_km', ...}, inplace=True)
    # 或者只保留需要的列
    # df = df[numeric_columns + 其它主键列]

    # 创建PCA双标图
    fig, ax, pca = create_pca_biplot(df)

    # 分析结果
    numeric_columns = [
        'road_length_km', 'car_count', 'ebike_count', 'bus_count', 
        'pedestrian_count', 'avg_speed_kph', 'traffic_flow_vph', 
        'traffic_density_vpkm', 'congestion_index'
    ]
    loadings_df = analyze_pca_results(pca, numeric_columns)

    # 显示图形
    plt.show()

    # 保存图片
    # fig.savefig('pca_biplot.png', dpi=300, bbox_inches='tight')
