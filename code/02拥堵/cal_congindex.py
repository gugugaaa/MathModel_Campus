import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_improved_congestion_index(df, method='enhanced'):
    """
    改进的拥挤指数计算方法
    
    Parameters:
    method: 'enhanced' | 'nonlinear' | 'adaptive' | 'original'
    """
    
    if method == 'enhanced':
        # 方案1：降低k_jam阈值，调整权重
        def kjam_enhanced(row):
            # 降低阈值，提高敏感性
            base = {1: 100, 2: 80, 3: 60, 4: 40}[row['road_type_code']]
            return base
        
        df['k_jam'] = df.apply(kjam_enhanced, axis=1)
        df['speed_factor'] = (1 - df['avg_speed_kph'] / df['JAM_SPEED']).clip(lower=0, upper=1)
        df['density_factor'] = (df['traffic_density_vpkm'] / df['k_jam']).clip(upper=1)
        
        # 调整权重，密度占更大比重
        α = 0.3  # 速度权重降低
        df['congestion_index'] = α * df['speed_factor'] + (1-α) * df['density_factor']
        
    elif method == 'nonlinear':
        # 方案2：非线性变换，使用平方根增强敏感性
        def kjam_original(row):
            base = {1: 150, 2: 120, 3: 100, 4: 70}[row['road_type_code']]
            return base
        
        df['k_jam'] = df.apply(kjam_original, axis=1)
        
        # 非线性变换
        speed_factor_raw = (1 - df['avg_speed_kph'] / df['JAM_SPEED']).clip(lower=0, upper=1)
        density_factor_raw = (df['traffic_density_vpkm'] / df['k_jam']).clip(upper=1)
        
        # 使用平方根变换增强低值区间的敏感性
        df['speed_factor'] = np.sqrt(speed_factor_raw)
        df['density_factor'] = np.sqrt(density_factor_raw)
        
        α = 0.4
        df['congestion_index'] = α * df['speed_factor'] + (1-α) * df['density_factor']
        
    elif method == 'adaptive':
        # 方案3：分道路类型自适应参数
        road_params = {
            1: {'k_jam': 120, 'speed_weight': 0.4, 'nonlinear_power': 0.7},  # 高速路
            2: {'k_jam': 90,  'speed_weight': 0.3, 'nonlinear_power': 0.8},  # 主干道  
            3: {'k_jam': 70,  'speed_weight': 0.3, 'nonlinear_power': 0.9},  # 次干道
            4: {'k_jam': 50,  'speed_weight': 0.2, 'nonlinear_power': 1.0}   # 支路
        }
        
        congestion_indices = []
        for _, row in df.iterrows():
            road_type = row['road_type_code']
            params = road_params[road_type]
            
            k_jam = params['k_jam']
            speed_weight = params['speed_weight']
            power = params['nonlinear_power']
            
            speed_factor = (1 - row['avg_speed_kph'] / row['JAM_SPEED'])
            speed_factor = max(0, min(1, speed_factor)) ** power
            
            density_factor = min(1, row['traffic_density_vpkm'] / k_jam) ** power
            
            cong_index = speed_weight * speed_factor + (1 - speed_weight) * density_factor
            congestion_indices.append(cong_index)
        
        df['congestion_index'] = congestion_indices
        
        # 重新计算因子用于分析
        df['k_jam'] = df['road_type_code'].map({k: v['k_jam'] for k, v in road_params.items()})
        df['speed_factor'] = [(1 - row['avg_speed_kph'] / row['JAM_SPEED']) for _, row in df.iterrows()]
        df['density_factor'] = [row['traffic_density_vpkm'] / row['k_jam'] for _, row in df.iterrows()]
        
    else:  # original
        # 原始方法
        def kjam_original(row):
            base = {1: 150, 2: 120, 3: 100, 4: 70}[row['road_type_code']]
            return base
        
        df['k_jam'] = df.apply(kjam_original, axis=1)
        df['speed_factor'] = (1 - df['avg_speed_kph'] / df['JAM_SPEED']).clip(lower=0, upper=1)
        df['density_factor'] = (df['traffic_density_vpkm'] / df['k_jam']).clip(upper=1)
        
        α = 0.5
        df['congestion_index'] = α * df['speed_factor'] + (1-α) * df['density_factor']
    
    return df

def compare_congestion_methods(df, sample_size=10000):
    """
    比较不同拥挤指数计算方法
    """
    methods = ['original', 'enhanced', 'nonlinear', 'adaptive']
    method_names = ['原始方法', '增强方法', '非线性方法', '自适应方法']
    
    # 抽样数据用于比较
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    results = {}
    
    print("不同方法计算结果对比：")
    print("="*70)
    
    for i, method in enumerate(methods):
        df_method = df_sample.copy()
        df_method = calculate_improved_congestion_index(df_method, method=method)
        
        # 添加拥挤等级
        def get_congestion_level(cong_index):
            if cong_index < 0.2:
                return '畅通'
            elif cong_index < 0.4:
                return '缓行' 
            elif cong_index < 0.6:
                return '拥挤'
            elif cong_index < 0.8:
                return '严重拥挤'
            else:
                return '极度拥挤'
        
        df_method['CongLevel'] = df_method['congestion_index'].apply(get_congestion_level)
        
        # 统计结果
        results[method] = {
            'mean': df_method['congestion_index'].mean(),
            'std': df_method['congestion_index'].std(),
            'median': df_method['congestion_index'].median(),
            'max': df_method['congestion_index'].max(),
            'level_dist': df_method['CongLevel'].value_counts(normalize=True),
            'road_type_mean': df_method.groupby('road_type_code')['congestion_index'].mean()
        }
        
        print(f"\n📊 {method_names[i]}:")
        print(f"   均值: {results[method]['mean']:.3f}")
        print(f"   标准差: {results[method]['std']:.3f}")  
        print(f"   中位数: {results[method]['median']:.3f}")
        print(f"   最大值: {results[method]['max']:.3f}")
        
        level_dist = results[method]['level_dist'] * 100
        print(f"   拥挤等级分布:")
        for level in ['畅通', '缓行', '拥挤', '严重拥挤', '极度拥挤']:
            if level in level_dist:
                print(f"     {level}: {level_dist[level]:.1f}%")
    
    return results

def visualize_method_comparison(df, sample_size=10000):
    """
    可视化不同方法的对比
    """
    methods = ['original', 'enhanced', 'nonlinear', 'adaptive']
    method_names = ['原始方法', '增强方法', '非线性方法', '自适应方法']
    colors = ['blue', 'green', 'orange', 'red']
    
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('不同拥挤指数计算方法对比', fontsize=16, fontweight='bold')
    
    # 存储各方法的结果
    method_results = {}
    
    for method in methods:
        df_method = df_sample.copy()
        df_method = calculate_improved_congestion_index(df_method, method=method)
        method_results[method] = df_method['congestion_index']
    
    # 1. 分布对比（直方图）
    for i, method in enumerate(methods):
        axes[0, 0].hist(method_results[method], bins=30, alpha=0.7, 
                       label=method_names[i], color=colors[i], density=True)
    axes[0, 0].set_title('拥挤指数分布对比')
    axes[0, 0].set_xlabel('拥挤指数')
    axes[0, 0].set_ylabel('密度')
    axes[0, 0].legend()
    
    # 2. 箱线图对比
    data_for_box = [method_results[method] for method in methods]
    box_plot = axes[0, 1].boxplot(data_for_box, labels=method_names, patch_artist=True)
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0, 1].set_title('拥挤指数箱线图对比')
    axes[0, 1].set_ylabel('拥挤指数')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. 累积分布函数对比
    for i, method in enumerate(methods):
        sorted_values = np.sort(method_results[method])
        cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        axes[1, 0].plot(sorted_values, cumulative, label=method_names[i], 
                       color=colors[i], linewidth=2)
    axes[1, 0].set_title('累积分布函数对比')
    axes[1, 0].set_xlabel('拥挤指数')
    axes[1, 0].set_ylabel('累积概率')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 统计量对比雷达图
    stats_data = []
    for method in methods:
        values = method_results[method]
        stats = [
            values.mean() * 5,  # 放大5倍便于显示
            values.std() * 10,  # 放大10倍便于显示
            (values > 0.2).mean() * 100,  # 非畅通比例
            (values > 0.4).mean() * 100,  # 拥挤及以上比例
            values.max() * 100  # 最大值百分比
        ]
        stats_data.append(stats)
    
    # 简化版雷达图（条形图替代）
    x_labels = ['均值×5', '标准差×10', '非畅通%', '拥挤%', '最大值%']
    x_pos = np.arange(len(x_labels))
    
    bar_width = 0.2
    for i, method in enumerate(methods):
        axes[1, 1].bar(x_pos + i * bar_width, stats_data[i], 
                      bar_width, label=method_names[i], color=colors[i], alpha=0.7)
    
    axes[1, 1].set_title('统计量对比')
    axes[1, 1].set_xlabel('统计指标')
    axes[1, 1].set_ylabel('数值')
    axes[1, 1].set_xticks(x_pos + bar_width * 1.5)
    axes[1, 1].set_xticklabels(x_labels, rotation=45)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()

def recommend_best_method(comparison_results):
    """
    推荐最佳方法
    """
    print("\n" + "="*50)
    print("           方法推荐分析")
    print("="*50)
    
    # 分析各方法的特点
    analyses = {}
    
    for method, results in comparison_results.items():
        mean_val = results['mean']
        std_val = results['std']
        max_val = results['max']
        
        # 计算非畅通比例
        non_smooth_ratio = 1 - results['level_dist'].get('畅通', 0)
        
        # 计算道路类型区分度（标准差）
        road_type_std = results['road_type_mean'].std()
        
        analyses[method] = {
            'discrimination': road_type_std,  # 区分度
            'sensitivity': std_val,  # 敏感性
            'coverage': non_smooth_ratio,  # 覆盖度
            'range_utilization': max_val  # 范围利用率
        }
    
    # 评分系统
    method_scores = {}
    method_names = {
        'original': '原始方法',
        'enhanced': '增强方法', 
        'nonlinear': '非线性方法',
        'adaptive': '自适应方法'
    }
    
    print("\n各方法评估:")
    for method, analysis in analyses.items():
        # 综合评分（归一化后加权）
        score = (
            analysis['discrimination'] * 0.3 +  # 区分度权重30%
            analysis['sensitivity'] * 0.25 +    # 敏感性权重25%
            analysis['coverage'] * 0.25 +       # 覆盖度权重25%
            analysis['range_utilization'] * 0.2  # 范围利用率权重20%
        )
        
        method_scores[method] = score
        
        print(f"\n🔍 {method_names[method]}:")
        print(f"   区分度: {analysis['discrimination']:.4f}")
        print(f"   敏感性: {analysis['sensitivity']:.4f}")  
        print(f"   覆盖度: {analysis['coverage']:.3f}")
        print(f"   范围利用率: {analysis['range_utilization']:.3f}")
        print(f"   综合评分: {score:.4f}")
    
    # 推荐最佳方法
    best_method = max(method_scores.keys(), key=lambda k: method_scores[k])
    
    print(f"\n🏆 推荐方法: {method_names[best_method]}")
    print(f"📈 推荐理由:")
    
    if best_method == 'enhanced':
        print("   - 通过降低k_jam阈值提高了敏感性")
        print("   - 调整权重突出密度因素的重要性")
        print("   - 计算简单，易于实现和理解")
    elif best_method == 'nonlinear':
        print("   - 非线性变换增强了低值区间的区分度")
        print("   - 更好地反映了拥挤程度的渐进性")
        print("   - 数学上更加合理")
    elif best_method == 'adaptive':
        print("   - 针对不同道路类型定制参数")
        print("   - 最大化各类道路的区分效果")
        print("   - 符合实际交通特征")
    else:
        print("   - 保持了原有的计算逻辑")
        print("   - 结果稳定可靠")
    
    return best_method

# 使用示例：
# results = compare_congestion_methods(df)
# visualize_method_comparison(df)
# best_method = recommend_best_method(results)

def apply_recommended_method(df, method='adaptive'):
    """
    应用推荐的方法重新计算拥挤指数
    """
    print(f"应用{method}方法重新计算拥挤指数...")
    
    df_new = calculate_improved_congestion_index(df.copy(), method=method)
    
    # 添加拥挤等级
    def get_congestion_level(cong_index):
        if cong_index < 0.2:
            return '畅通'
        elif cong_index < 0.4:
            return '缓行'
        elif cong_index < 0.6:
            return '拥挤'
        elif cong_index < 0.8:
            return '严重拥挤'
        else:
            return '极度拥挤'
    
    df_new['CongLevel'] = df_new['congestion_index'].apply(get_congestion_level)
    
    return df_new

def analyze_and_export_congestion_data():
    """
    分析速度属性数据并导出拥挤指数结果
    """
    # 读取数据
    input_file = r"data\02重命名.csv"
    output_file = r"data\03拥挤.csv"
    
    print(f"正在读取数据: {input_file}")
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(input_file, encoding='gbk')
    
    print(f"数据shape: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # 检查必要的列是否存在
    required_columns = ['avg_speed_kph', 'traffic_density_vpkm', 'road_type_code']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"警告: 缺少必要列 {missing_columns}")
        print("请确认数据文件包含以下列: avg_speed_kph, traffic_density_vpkm, road_type_code")
        return
    
    # 根据道路类型生成JAM_SPEED
    print("\n正在根据道路类型生成JAM_SPEED...")
    def get_jam_speed(road_type):
        jam_speed_map = {1: 30, 2: 30, 3: 20, 4: 15}
        return jam_speed_map.get(road_type, 20)  # 默认值20
    
    df['JAM_SPEED'] = df['road_type_code'].apply(get_jam_speed)
    print(f"已生成JAM_SPEED列，各道路类型对应速度:")
    for road_type in sorted(df['road_type_code'].unique()):
        jam_speed = get_jam_speed(road_type)
        print(f"  道路类型{road_type}: {jam_speed}km/h")
    
    # 数据预处理
    print("\n正在进行数据预处理...")
    required_columns.append('JAM_SPEED')  # 添加JAM_SPEED到必要列
    df = df.dropna(subset=required_columns)
    print(f"去除缺失值后数据shape: {df.shape}")
    
    # 比较不同方法
    print("\n开始比较不同拥挤指数计算方法...")
    results = compare_congestion_methods(df, sample_size=min(10000, len(df)))
    
    # 可视化对比
    print("\n正在生成对比图表...")
    visualize_method_comparison(df, sample_size=min(10000, len(df)))
    
    # 推荐最佳方法
    print("\n正在分析推荐最佳方法...")
    best_method = recommend_best_method(results)
    
    # 应用推荐方法计算全部数据
    print(f"\n正在应用{best_method}方法计算全部数据的拥挤指数...")
    df_final = apply_recommended_method(df, method=best_method)
    
    # 输出统计信息
    print("\n📊 最终结果统计:")
    print(f"总记录数: {len(df_final)}")
    print(f"拥挤指数统计:")
    print(f"  均值: {df_final['congestion_index'].mean():.3f}")
    print(f"  标准差: {df_final['congestion_index'].std():.3f}")
    print(f"  最小值: {df_final['congestion_index'].min():.3f}")
    print(f"  最大值: {df_final['congestion_index'].max():.3f}")
    
    print(f"\n拥挤等级分布:")
    level_counts = df_final['CongLevel'].value_counts()
    for level in ['畅通', '缓行', '拥挤', '严重拥挤', '极度拥挤']:
        if level in level_counts:
            count = level_counts[level]
            percentage = count / len(df_final) * 100
            print(f"  {level}: {count} ({percentage:.1f}%)")
    
    print(f"\n各道路类型拥挤指数均值:")
    road_type_stats = df_final.groupby('road_type_code')['congestion_index'].agg(['mean', 'std', 'count'])
    for road_type in sorted(df_final['road_type_code'].unique()):
        stats = road_type_stats.loc[road_type]
        print(f"  道路类型{road_type}: 均值={stats['mean']:.3f}, 标准差={stats['std']:.3f}, 记录数={stats['count']}")
    
    # 导出结果
    print(f"\n正在导出结果到: {output_file}")
    df_final.to_csv(output_file, index=False, encoding='utf-8-sig')
    print("✅ 导出完成!")
    
    return df_final

if __name__ == "__main__":
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 执行分析
    result_df = analyze_and_export_congestion_data()