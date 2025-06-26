import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体，确保图表能正确显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 固定参数 ---
# 文件路径
input_file = r'data\02重命名.csv'
# 载人量计算的固定参数
P_car_avg = 1.3    # 平均每辆小汽车载客人数
P_ebike_avg = 1.1  # 平均每辆电动车载客人数
P_bus_avg = 60     # 平均每辆公交车载客人数

def load_and_prepare_data():
    """
    加载并预处理数据。
    - 读取CSV文件。
    - 根据车辆数计算总载人量。
    - 添加易于理解的道路类型名称。
    - 转换时间格式，便于按时间顺序绘图。
    """
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(input_file, encoding='gbk')
    
    # 打印数据基本信息，用于调试
    print(f"原始数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    
    # 核心计算：根据不同交通工具的数量和平均载客数，计算总载人量
    df['passenger_throughput'] = (
        P_car_avg * df['car_count'] +
        P_ebike_avg * df['ebike_count'] +
        P_bus_avg * df['bus_count']
    )
    
    # 数据清洗与转换：将道路类型编码映射为可读的名称
    road_type_names = {1: '高速公路', 2: '主干道', 3: '次干道', 4: '支路'}
    df['road_type_name'] = df['road_type_code'].map(road_type_names)
    
    # 将 'time_slot_30min' 转换为可排序的数值（例如，'08:30' -> 8.5）
    # 这对于绘制按时间顺序排列的折线图至关重要
    if 'time_slot_30min' in df.columns:
        df['time_hour'] = df['time_slot_30min'].apply(lambda x: float(x.split(':')[0]) + float(x.split(':')[1])/60)
    else:
        # 如果没有这个列，就创建一个默认的，避免后续代码出错
        df['time_hour'] = 0 

    return df

def create_focused_dashboard(df):
    """
    创建一站式可视化仪表板 (2x2布局)。
    这张仪表板包含了四个最有说服力的核心图表。
    """
    
    # 创建一个 2x2 的图纸(figure)和子图(axes)
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('城市交通载人量核心分析仪表板', fontsize=20, fontweight='bold')

    # --- 图1: 载人量时间变化趋势 (折线图) ---
    # 目的：展示一天内不同类型道路的载人量高峰和低谷。
    ax1 = axes[0, 0]
    # 按小时和道路类型对载人量取平均值
    time_avg = df.groupby(['time_hour', 'road_type_name'])['passenger_throughput'].mean().unstack()
    time_avg.plot(kind='line', marker='o', linewidth=2, ax=ax1, colormap='viridis')
    ax1.set_title('各类型道路载人量24小时变化趋势', fontsize=14)
    ax1.set_xlabel('时间（小时）', fontsize=12)
    ax1.set_ylabel('平均载人量', fontsize=12)
    ax1.legend(title='道路类型')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- 图2: 各道路类型总载人量对比 (条形图) ---
    # 目的：清晰比较不同类型道路在总载人量上的贡献大小。
    ax2 = axes[0, 1]
    road_totals = df.groupby('road_type_name')['passenger_throughput'].sum().sort_values()
    road_totals.plot(kind='barh', ax=ax2, color=sns.color_palette("viridis", len(road_totals)))
    ax2.set_title('各类型道路总载人量贡献对比', fontsize=14)
    ax2.set_xlabel('总载人量 (人次)', fontsize=12)
    ax2.set_ylabel('道路类型', fontsize=12)
    # 在条形图上显示具体数值
    for index, value in enumerate(road_totals):
        ax2.text(value, index, f' {value:,.0f}', va='center')

    # --- 图3: 各道路类型载人量分布 (箱线图) ---
    # 目的：展示每种道路类型载人量的中位数、波动范围和异常值。
    ax3 = axes[1, 0]
    sns.boxplot(data=df, x='road_type_name', y='passenger_throughput', ax=ax3, palette='viridis')
    ax3.set_title('各类型道路载人量分布', fontsize=14)
    ax3.set_xlabel('道路类型', fontsize=12)
    ax3.set_ylabel('载人量', fontsize=12)
    ax3.tick_params(axis='x', rotation=15)

    # --- 图4: 载人量热力图 ---
    # 目的：结合时间和道路类型，精确定位最繁忙的时段和路段类型。
    ax4 = axes[1, 1]
    if 'time_slot_30min' in df.columns:
        # 创建数据透视表，行为道路类型，列为时间段
        pivot_data = df.groupby(['road_type_name', 'time_slot_30min'])['passenger_throughput'].mean().unstack()
        # 重新排序时间列，使其从0点到23:30
        time_cols = sorted(pivot_data.columns, key=lambda x: (int(x.split(':')[0]), int(x.split(':')[1])))
        pivot_data = pivot_data[time_cols]
        sns.heatmap(pivot_data, annot=False, cmap='YlOrRd', ax=ax4)
        ax4.set_title('各道路类型在不同时间段的载人量热力图', fontsize=14)
        ax4.set_xlabel('时间段', fontsize=12)
        ax4.set_ylabel('道路类型', fontsize=12)
    else:
        ax4.text(0.5, 0.5, '缺少 time_slot_30min 数据', ha='center', va='center')
        ax4.set_title('热力图数据缺失', fontsize=14)
    
    # 调整布局，防止标题和标签重叠
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # 显示最终图表
    plt.show()

def analyze_peak_hours_summary(df):
    """
    在控制台打印核心统计摘要。
    这部分对于快速获取关键数字很有用。
    """
    print("\n" + "="*20 + " 核心数据摘要 " + "="*20)
    
    # 总体统计
    print(f"数据总条目数: {len(df)}")
    print(f"总计载人量: {df['passenger_throughput'].sum():,.0f} 人次")
    
    # 按道路类型统计最高载人量的时间
    print("\n--- 各道路类型高峰时段分析 ---")
    peak_times = df.groupby(['road_type_name', 'time_slot_30min'])['passenger_throughput'].mean()
    for road_type in df['road_type_name'].unique():
        if road_type is not None:
            # 找到当前道路类型的顶峰时间和数值
            peak_info = peak_times.loc[road_type].idxmax()
            peak_value = peak_times.loc[road_type].max()
            print(f"【{road_type}】的高峰时段是: {peak_info} (平均载人量: {peak_value:.2f})")
    
    print("\n" + "="*50)


if __name__ == "__main__":
    # 步骤1：加载和准备数据
    traffic_df = load_and_prepare_data()
    
    # 步骤2：在控制台打印关键分析数据
    analyze_peak_hours_summary(traffic_df)
    
    # 步骤3：创建并显示精简后的可视化仪表板
    create_focused_dashboard(traffic_df)
    
    print("\n可视化仪表板已生成！")