import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def create_pca_biplot(df):
    """
    创建PCA双标图，特别标记安全指数和相关影响因素
    """
    # 选择数值型变量进行PCA分析（去除载客量和CO2排放）
    numeric_columns = [
        'road_length_km', 'car_count', 'ebike_count', 'bus_count', 
        'pedestrian_count', 'avg_speed_kph', 'traffic_flow_vph', 
        'traffic_density_vpkm', 'congestion_index', 'safety_index'
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
    
    # 绘制变量向量和卡片文本
    texts = []
    text_positions = []
    for i, var in enumerate(numeric_columns):
        # 配色方案
        if var == 'safety_index':
            color = '#FF0000'  # 红色
            linewidth = 3.0
            alpha = 0.9
        elif var in ['congestion_index', 'traffic_density_vpkm', 'avg_speed_kph', 'traffic_flow_vph']:
            color = '#FFD700'  # 黄
            linewidth = 2.5
            alpha = 0.8
        elif var == 'ebike_count':
            color = '#0066FF'  # 蓝
            linewidth = 2.5
            alpha = 0.85
        elif var in ['car_count', 'bus_count', 'pedestrian_count']:
            color = '#228B22'  # 绿
            linewidth = 2.0
            alpha = 0.75
        else:  # road_length_km
            color = '#696969'  # 灰
            linewidth = 1.8
            alpha = 0.7
        
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
            'safety_index': '安全指数'
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
        
        # 卡片样式文本框
        fontweight = 'bold' if var in ['safety_index', 'congestion_index', 'avg_speed_kph', 'traffic_density_vpkm'] else 'normal'
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
    ax.set_title(f'道路安全数据PCA双标图\n累计解释方差: {sum(pca.explained_variance_ratio_):.2%}', 
                fontsize=14, fontweight='bold')
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 添加原点线
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#FF0000', lw=3, label='安全指数'),
        Line2D([0], [0], color='#FFD700', lw=2.5, label='交通指标'),
        Line2D([0], [0], color='#0066FF', lw=2.5, label='电动车数量'),
        Line2D([0], [0], color='#228B22', lw=2, label='其他车辆类型'),
        Line2D([0], [0], color='#696969', lw=1.8, label='道路参数')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # 添加解释性文本
    ax.text(0.02, 0.98, '安全指数与风险因素呈负相关\n(方向相反是正常现象)', 
            transform=ax.transAxes, fontsize=9, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
            verticalalignment='top')
    
    plt.tight_layout()
    return fig, ax, pca

def analyze_pca_results(pca, numeric_columns):
    """
    分析PCA结果，重点关注安全指数
    """
    print("=== 安全指数PCA分析结果 ===")
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
    
    # 安全指数相关性分析
    print("\n=== 安全指数载荷分析 ===")
    safety_pc1 = loadings_df.loc['safety_index', 'PC1']
    safety_pc2 = loadings_df.loc['safety_index', 'PC2']
    print(f"安全指数在PC1上的载荷: {safety_pc1:.3f}")
    print(f"安全指数在PC2上的载荷: {safety_pc2:.3f}")
    
    # 分析安全指数与风险因素的关系
    print(f"\n=== 安全指数与风险因素关系分析 ===")
    risk_factors = ['congestion_index', 'traffic_density_vpkm', 'car_count', 'ebike_count']
    print("注意: 负相关表示安全指数高时风险因素低(这是合理的)")
    
    for var in risk_factors:
        if var in numeric_columns:
            pc1_load = loadings_df.loc[var, 'PC1']
            pc2_load = loadings_df.loc[var, 'PC2']
            # 计算向量夹角的余弦值
            dot_product = safety_pc1 * pc1_load + safety_pc2 * pc2_load
            magnitude_safety = np.sqrt(safety_pc1**2 + safety_pc2**2)
            magnitude_var = np.sqrt(pc1_load**2 + pc2_load**2)
            cosine_similarity = dot_product / (magnitude_safety * magnitude_var)
            
            correlation_type = "正相关" if cosine_similarity > 0 else "负相关"
            print(f"  安全指数 vs {var}: {correlation_type} (相似度: {cosine_similarity:.3f})")
    
    # 找出与安全指数方向相似的有益因素
    print(f"\n与安全指数方向相似的变量(有益因素):")
    for var in numeric_columns:
        if var != 'safety_index':
            pc1_load = loadings_df.loc[var, 'PC1']
            pc2_load = loadings_df.loc[var, 'PC2']
            dot_product = safety_pc1 * pc1_load + safety_pc2 * pc2_load
            magnitude_safety = np.sqrt(safety_pc1**2 + safety_pc2**2)
            magnitude_var = np.sqrt(pc1_load**2 + pc2_load**2)
            cosine_similarity = dot_product / (magnitude_safety * magnitude_var)
            
            if cosine_similarity > 0.3:  # 正相关
                print(f"  {var}: 相似度 {cosine_similarity:.3f}")
    
    return loadings_df

# 示例使用代码
if __name__ == "__main__":
    # 读取安全数据
    csv_path = r"E:\github_projects\math_modeling\data\07安全.csv"
    df = pd.read_csv(csv_path, encoding='utf-8')  # 如有乱码可尝试 encoding='gbk'

    # 创建PCA双标图
    fig, ax, pca = create_pca_biplot(df)

    # 分析结果
    numeric_columns = [
        'road_length_km', 'car_count', 'ebike_count', 'bus_count', 
        'pedestrian_count', 'avg_speed_kph', 'traffic_flow_vph', 
        'traffic_density_vpkm', 'congestion_index', 'safety_index'
    ]
    loadings_df = analyze_pca_results(pca, numeric_columns)

    # 显示图形
    plt.show()

    # 保存图片
    # fig.savefig('safety_pca_biplot.png', dpi=300, bbox_inches='tight')
