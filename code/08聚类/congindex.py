import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# 设置matplotlib支持中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

def analyze_traffic_congestion(filename):
    """
    使用K-means聚类按拥堵指数将交通数据分成三类：高、中、低拥堵
    
    参数:
    filename: str - CSV数据文件路径
    
    返回:
    dict - 包含聚类结果和统计信息的字典
    """
    
    try:
        # 1. 读取数据
        print(f"正在读取文件: {filename}")
        df = pd.read_csv(filename, encoding='utf-8')
        
        print(f"\n数据加载完成:")
        print(f"- 总行数: {len(df)}")
        print(f"- 列数: {len(df.columns)}")
        print(f"- 字段名: {', '.join(df.columns.tolist())}")
        
        # 2. 检查拥堵指数字段
        if 'congestion_index' not in df.columns:
            raise ValueError('数据中没有找到 congestion_index 字段')
        
        # 3. 数据预处理 - 过滤无效的拥堵指数值
        valid_mask = df['congestion_index'].notna() & np.isfinite(df['congestion_index'])
        df_valid = df[valid_mask].copy()
        
        print(f"\n拥堵指数统计:")
        print(f"- 有效数据点: {len(df_valid)}")
        print(f"- 无效数据点: {len(df) - len(df_valid)}")
        print(f"- 最小值: {df_valid['congestion_index'].min():.4f}")
        print(f"- 最大值: {df_valid['congestion_index'].max():.4f}")
        print(f"- 平均值: {df_valid['congestion_index'].mean():.4f}")
        print(f"- 标准差: {df_valid['congestion_index'].std():.4f}")
        
        # 4. 执行K-means聚类
        print(f"\n开始执行K-means聚类...")
        
        # 准备聚类数据 (reshape为2D数组)
        X = df_valid['congestion_index'].values.reshape(-1, 1)
        
        # 设置随机种子确保结果可重现
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10, max_iter=300)
        cluster_labels = kmeans.fit_predict(X)
        
        # 获取聚类中心
        cluster_centers = kmeans.cluster_centers_.flatten()
        
        # 5. 按聚类中心大小排序，确保标签对应：0=低拥堵，1=中拥堵，2=高拥堵
        center_order = np.argsort(cluster_centers)
        label_mapping = {center_order[i]: i for i in range(3)}
        
        # 重新映射标签
        remapped_labels = np.array([label_mapping[label] for label in cluster_labels])
        sorted_centers = np.sort(cluster_centers)
        
        # 6. 添加聚类标签到数据框
        df_valid = df_valid.copy()
        df_valid['cluster_label'] = remapped_labels
        df_valid['cluster_name'] = df_valid['cluster_label'].map({
            0: '低拥堵', 
            1: '中拥堵', 
            2: '高拥堵'
        })
        
        print(f"\n聚类结果:")
        for i, (label, center) in enumerate(zip(['低拥堵', '中拥堵', '高拥堵'], sorted_centers)):
            count = np.sum(remapped_labels == i)
            percentage = count / len(remapped_labels) * 100
            print(f"- {label}: 中心={center:.4f}, 数量={count} ({percentage:.1f}%)")
        
        # 7. 按聚类结果分组
        clustered_data = {}
        cluster_stats = {}
        
        for cluster_id, cluster_name in [(0, '低拥堵'), (1, '中拥堵'), (2, '高拥堵')]:
            cluster_df = df_valid[df_valid['cluster_label'] == cluster_id].copy()
            
            # 移除聚类标签列用于输出
            output_df = cluster_df.drop(['cluster_label', 'cluster_name'], axis=1)
            clustered_data[cluster_name] = output_df
            
            # 计算统计信息
            if len(cluster_df) > 0:
                congestion_values = cluster_df['congestion_index']
                cluster_stats[cluster_name] = {
                    'count': len(cluster_df),
                    'min': congestion_values.min(),
                    'max': congestion_values.max(),
                    'mean': congestion_values.mean(),
                    'std': congestion_values.std()
                }
        
        # 8. 显示各聚类的详细统计
        print(f"\n各聚类拥堵指数详细统计:")
        for cluster_name, stats in cluster_stats.items():
            print(f"{cluster_name}:")
            print(f"  - 数量: {stats['count']}")
            print(f"  - 范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"  - 平均值: {stats['mean']:.4f}")
            print(f"  - 标准差: {stats['std']:.4f}")
        
        # 9. 保存分类后的数据文件
        print(f"\n正在保存聚类结果文件...")
        
        for cluster_name, cluster_df in clustered_data.items():
            output_filename = f"{cluster_name}_路段数据.csv"
            cluster_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
            print(f"- 已保存: {output_filename} ({len(cluster_df)} 条记录)")
        
        # 10. 可视化聚类结果
        print(f"\n生成聚类可视化图表...")
        
        plt.figure(figsize=(15, 5))
        
        # 子图1: 拥堵指数分布直方图
        plt.subplot(1, 3, 1)
        colors = ['green', 'orange', 'red']
        for i, (cluster_name, color) in enumerate(zip(['低拥堵', '中拥堵', '高拥堵'], colors)):
            cluster_data = df_valid[df_valid['cluster_label'] == i]['congestion_index']
            plt.hist(cluster_data, bins=20, alpha=0.7, label=cluster_name, color=color)
        
        plt.xlabel('拥堵指数')
        plt.ylabel('频次')
        plt.title('拥堵指数分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2: 聚类结果散点图
        plt.subplot(1, 3, 2)
        for i, (cluster_name, color) in enumerate(zip(['低拥堵', '中拥堵', '高拥堵'], colors)):
            cluster_data = df_valid[df_valid['cluster_label'] == i]
            plt.scatter(range(len(cluster_data)), 
                       cluster_data['congestion_index'], 
                       c=color, label=cluster_name, alpha=0.6, s=10)
        
        plt.xlabel('数据点索引')
        plt.ylabel('拥堵指数')
        plt.title('聚类结果')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图3: 聚类比例饼图
        plt.subplot(1, 3, 3)
        cluster_counts = [cluster_stats[name]['count'] for name in ['低拥堵', '中拥堵', '高拥堵']]
        plt.pie(cluster_counts, labels=['低拥堵', '中拥堵', '高拥堵'], 
                colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('聚类比例分布')
        
        plt.tight_layout()
        plt.savefig('拥堵指数聚类分析结果.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 11. 返回结果
        return {
            'summary': {
                'total_records': len(df),
                'valid_records': len(df_valid),
                'invalid_records': len(df) - len(df_valid),
                'cluster_centers': sorted_centers.tolist(),
                'cluster_counts': cluster_counts,
                'cluster_stats': cluster_stats
            },
            'clustered_data': clustered_data,
            'full_data_with_labels': df_valid
        }
        
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        raise e

def analyze_cluster_features(clustered_data, feature_columns=None):
    """
    分析各聚类在不同特征上的差异
    
    参数:
    clustered_data: dict - 聚类后的数据字典
    feature_columns: list - 要分析的特征列名列表
    """
    
    if feature_columns is None:
        # 默认分析的特征
        feature_columns = ['car_count', 'ebike_count', 'bus_count', 'pedestrian_count', 
                          'avg_speed_kph', 'traffic_flow_vph', 'traffic_density_vpkm']
    
    print("\n=== 各聚类特征对比分析 ===")
    
    # 创建特征对比表
    comparison_data = []
    
    for cluster_name, cluster_df in clustered_data.items():
        row = {'聚类': cluster_name}
        for feature in feature_columns:
            if feature in cluster_df.columns:
                row[f'{feature}_mean'] = cluster_df[feature].mean()
                row[f'{feature}_std'] = cluster_df[feature].std()
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # 显示特征均值对比
    print("\n各聚类主要特征均值对比:")
    for feature in feature_columns:
        if f'{feature}_mean' in comparison_df.columns:
            print(f"\n{feature}:")
            for _, row in comparison_df.iterrows():
                print(f"  {row['聚类']}: {row[f'{feature}_mean']:.2f} (±{row[f'{feature}_std']:.2f})")
    
    return comparison_df

# 主函数调用示例
if __name__ == "__main__":
    print("=== 拥堵指数聚类分析工具 ===")
    print("请确保您的数据文件包含 'congestion_index' 字段")
    
    # 使用示例：
    filename = r"E:\github_projects\math_modeling\data\07安全.csv"  # 替换为您的实际文件名
    result = analyze_traffic_congestion(filename)
    # 可选：分析各聚类的特征差异
    feature_analysis = analyze_cluster_features(result['clustered_data'])
