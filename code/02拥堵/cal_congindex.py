import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_congestion_index(df):
    """
    使用非线性方法计算拥挤指数
    
    特点：
    - 使用平方根变换增强低值区间的敏感性
    - 非线性变换使拥挤程度的变化更加平滑和合理
    - 速度因子权重为0.4，密度因子权重为0.6
    """
    
    # 设置各道路类型的饱和密度k_jam
    def get_kjam(road_type):
        k_jam_map = {1: 150, 2: 120, 3: 100, 4: 70}
        return k_jam_map.get(road_type, 70)
    
    df['k_jam'] = df['road_type_code'].apply(get_kjam)
    
    # 计算原始因子
    speed_factor_raw = (1 - df['avg_speed_kph'] / df['JAM_SPEED']).clip(lower=0, upper=1)
    density_factor_raw = (df['traffic_density_vpkm'] / df['k_jam']).clip(upper=1)
    
    # 非线性变换：使用平方根增强低值区间的敏感性
    df['speed_factor'] = np.sqrt(speed_factor_raw)
    df['density_factor'] = np.sqrt(density_factor_raw)
    
    # 计算拥挤指数（速度因子权重0.4，密度因子权重0.6）
    α = 0.4
    df['congestion_index'] = α * df['speed_factor'] + (1-α) * df['density_factor']
    
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
    
    df['CongLevel'] = df['congestion_index'].apply(get_congestion_level)
    
    return df

def analyze_congestion_results(df):
    """
    分析拥挤指数计算结果
    """
    print("📊 拥挤指数计算结果分析")
    print("="*50)
    
    # 基础统计
    print(f"\n基础统计信息:")
    print(f"  总记录数: {len(df)}")
    print(f"  拥挤指数均值: {df['congestion_index'].mean():.3f}")
    print(f"  拥挤指数标准差: {df['congestion_index'].std():.3f}")
    print(f"  拥挤指数中位数: {df['congestion_index'].median():.3f}")
    print(f"  拥挤指数最大值: {df['congestion_index'].max():.3f}")
    print(f"  拥挤指数最小值: {df['congestion_index'].min():.3f}")
    
    # 拥挤等级分布
    print(f"\n拥挤等级分布:")
    level_counts = df['CongLevel'].value_counts()
    for level in ['畅通', '缓行', '拥挤', '严重拥挤', '极度拥挤']:
        if level in level_counts:
            count = level_counts[level]
            percentage = count / len(df) * 100
            print(f"  {level}: {count:,} ({percentage:.1f}%)")
    
    # 各道路类型统计
    print(f"\n各道路类型拥挤指数统计:")
    road_type_stats = df.groupby('road_type_code')['congestion_index'].agg(['mean', 'std', 'count'])
    for road_type in sorted(df['road_type_code'].unique()):
        stats = road_type_stats.loc[road_type]
        print(f"  道路类型{road_type}: 均值={stats['mean']:.3f}, 标准差={stats['std']:.3f}, 记录数={stats['count']:,}")
    
    return level_counts, road_type_stats

def visualize_congestion_results(df, sample_size=10000):
    """
    可视化拥挤指数计算结果
    """
    # 抽样用于可视化
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('拥挤指数计算结果分析（非线性方法）', fontsize=16, fontweight='bold')
    
    # 1. 拥挤指数分布直方图
    axes[0, 0].hist(df_sample['congestion_index'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0, 0].set_title('拥挤指数分布')
    axes[0, 0].set_xlabel('拥挤指数')
    axes[0, 0].set_ylabel('频数')
    axes[0, 0].axvline(df_sample['congestion_index'].mean(), color='red', linestyle='--', 
                       label=f'均值: {df_sample["congestion_index"].mean():.3f}')
    axes[0, 0].legend()
    
    # 2. 各道路类型拥挤指数箱线图
    road_types = sorted(df_sample['road_type_code'].unique())
    congestion_by_road = [df_sample[df_sample['road_type_code']==rt]['congestion_index'] for rt in road_types]
    
    box_plot = axes[0, 1].boxplot(congestion_by_road, labels=[f'类型{rt}' for rt in road_types], 
                                  patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'orange', 'pink']
    for patch, color in zip(box_plot['boxes'], colors[:len(road_types)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0, 1].set_title('各道路类型拥挤指数分布')
    axes[0, 1].set_ylabel('拥挤指数')
    
    # 3. 拥挤等级分布饼图
    level_counts = df_sample['CongLevel'].value_counts()
    colors_pie = ['green', 'yellow', 'orange', 'red', 'darkred']
    axes[1, 0].pie(level_counts.values, labels=level_counts.index, autopct='%1.1f%%', 
                   colors=colors_pie[:len(level_counts)], startangle=90)
    axes[1, 0].set_title('拥挤等级分布')
    
    # 4. 速度因子 vs 密度因子散点图
    scatter = axes[1, 1].scatter(df_sample['speed_factor'], df_sample['density_factor'], 
                                c=df_sample['congestion_index'], cmap='viridis', alpha=0.6, s=20)
    axes[1, 1].set_xlabel('速度因子')
    axes[1, 1].set_ylabel('密度因子')
    axes[1, 1].set_title('速度因子 vs 密度因子')
    plt.colorbar(scatter, ax=axes[1, 1], label='拥挤指数')
    
    plt.tight_layout()
    plt.show()

def process_congestion_data():
    """
    处理拥挤指数数据的主函数
    """
    # 文件路径
    input_file = r"data\02重命名.csv"
    output_file = r"data\03拥挤.csv"
    
    print("🚀 开始处理拥挤指数数据")
    print("="*50)
    
    # 读取数据
    print(f"正在读取数据: {input_file}")
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(input_file, encoding='gbk')
        except Exception as e:
            print(f"❌ 读取文件失败: {e}")
            return None
    
    print(f"✅ 数据读取成功，shape: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # 检查必要的列
    required_columns = ['avg_speed_kph', 'road_type_code']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"❌ 缺少必要列: {missing_columns}")
        print("请确认数据文件包含以下列: avg_speed_kph, road_type_code")
        return None
    
    # 生成JAM_SPEED（各道路类型的阻塞速度）
    print("\n正在生成JAM_SPEED...")
    def get_jam_speed(road_type):
        jam_speed_map = {1: 30, 2: 30, 3: 20, 4: 15}  # 高速路、主干道、次干道、支路
        return jam_speed_map.get(road_type, 20)
    
    df['JAM_SPEED'] = df['road_type_code'].apply(get_jam_speed)
    
    # 处理traffic_density_vpkm
    if 'traffic_density_vpkm' not in df.columns:
        print("⚠️ 缺少traffic_density_vpkm，正在生成模拟数据...")
        # 基于速度和道路类型生成合理的交通密度
        np.random.seed(42)
        base_density = (50 - df['avg_speed_kph']) * df['road_type_code'] * 1.5
        noise = np.random.normal(0, 5, len(df))
        df['traffic_density_vpkm'] = (base_density + noise).clip(lower=0)
        print("✅ 模拟交通密度数据生成完成")
    
    # 数据清洗
    print("\n正在进行数据清洗...")
    original_count = len(df)
    df = df.dropna(subset=required_columns + ['JAM_SPEED', 'traffic_density_vpkm'])
    cleaned_count = len(df)
    print(f"数据清洗完成: {original_count} -> {cleaned_count} (移除了{original_count-cleaned_count}条记录)")
    
    if cleaned_count == 0:
        print("❌ 清洗后数据为空，请检查数据质量")
        return None
    
    # 计算拥挤指数
    print("\n正在计算拥挤指数（非线性方法）...")
    df = calculate_congestion_index(df)
    print("✅ 拥挤指数计算完成")
    
    # 分析结果
    print("\n" + "="*50)
    level_counts, road_type_stats = analyze_congestion_results(df)
    
    # 可视化结果
    print("\n正在生成可视化图表...")
    visualize_congestion_results(df)
    
    # 导出结果
    print(f"\n正在导出结果到: {output_file}")
    try:
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print("✅ 数据导出成功!")
    except Exception as e:
        print(f"❌ 导出失败: {e}")
        return None
    
    print("\n🎉 拥挤指数计算完成!")
    print(f"最终数据包含 {len(df)} 条记录")
    print(f"拥挤指数范围: {df['congestion_index'].min():.3f} - {df['congestion_index'].max():.3f}")
    
    return df

if __name__ == "__main__":
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 执行拥挤指数计算
    result_df = process_congestion_data()
    
    if result_df is not None:
        print(f"\n✨ 处理完成！结果已保存到 data\\03拥挤.csv")
    else:
        print("\n❌ 处理失败，请检查数据和参数设置")
