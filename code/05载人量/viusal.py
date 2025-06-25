import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 文件路径
input_file = r'data\02重命名.csv'

# 载人量计算的固定参数（来自create.py）
P_car_avg = 1.3    # 平均每辆小汽车载客人数
P_ebike_avg = 1.1  # 平均每辆电动车载客人数
P_bus_avg = 60     # 平均每辆公交车载客人数

def load_and_calculate_data():
    """加载数据并计算载人量"""
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(input_file, encoding='gbk')
    
    print(f"原始数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    
    # 计算载人量
    df['passenger_throughput'] = (
        P_car_avg * df['car_count'] +
        P_ebike_avg * df['ebike_count'] +
        P_bus_avg * df['bus_count']
    )
    
    # 添加道路类型名称
    road_type_names = {1: '高速公路', 2: '主干道', 3: '次干道', 4: '支路'}
    df['road_type_name'] = df['road_type_code'].map(road_type_names)
    
    # 转换时间格式便于排序
    df['time_hour'] = df['time_slot_30min'].apply(lambda x: float(x.split(':')[0]) + float(x.split(':')[1])/60)
    
    return df

def visualize_passenger_throughput(df):
    """可视化载人量数据"""
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('不同道路类型在各时间段的载人量分析', fontsize=16, fontweight='bold')
    
    # 1. 按道路类型和时间段的载人量热力图
    pivot_data = df.groupby(['road_type_name', 'time_slot_30min'])['passenger_throughput'].mean().unstack()
    sns.heatmap(pivot_data, annot=False, cmap='YlOrRd', ax=axes[0,0])
    axes[0,0].set_title('各道路类型时间段载人量热力图')
    axes[0,0].set_xlabel('时间段')
    axes[0,0].set_ylabel('道路类型')
    
    # 2. 各道路类型载人量箱线图
    sns.boxplot(data=df, x='road_type_name', y='passenger_throughput', ax=axes[0,1])
    axes[0,1].set_title('各道路类型载人量分布')
    axes[0,1].set_xlabel('道路类型')
    axes[0,1].set_ylabel('载人量')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. 时间段变化趋势
    time_avg = df.groupby(['time_hour', 'road_type_name'])['passenger_throughput'].mean().reset_index()
    for road_type in time_avg['road_type_name'].unique():
        data = time_avg[time_avg['road_type_name'] == road_type]
        axes[1,0].plot(data['time_hour'], data['passenger_throughput'], 
                      marker='o', label=road_type, linewidth=2)
    
    axes[1,0].set_title('载人量时间变化趋势')
    axes[1,0].set_xlabel('时间（小时）')
    axes[1,0].set_ylabel('平均载人量')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. 各道路类型载人量占比
    road_totals = df.groupby('road_type_name')['passenger_throughput'].sum()
    axes[1,1].pie(road_totals.values, labels=road_totals.index, autopct='%1.1f%%', startangle=90)
    axes[1,1].set_title('各道路类型载人量占比')
    
    plt.tight_layout()
    plt.show()

def analyze_peak_hours(df):
    """分析高峰时段"""
    print("\n=== 载人量统计分析 ===")
    
    # 总体统计
    print(f"总载人量: {df['passenger_throughput'].sum():,.0f}")
    print(f"平均载人量: {df['passenger_throughput'].mean():.2f}")
    print(f"载人量标准差: {df['passenger_throughput'].std():.2f}")
    
    # 按道路类型统计
    print("\n按道路类型统计:")
    road_stats = df.groupby('road_type_name')['passenger_throughput'].agg(['sum', 'mean', 'max', 'count'])
    print(road_stats.round(2))
    
    # 找出载人量最高的时间段
    print("\n各道路类型载人量最高的时间段:")
    for road_type in df['road_type_name'].unique():
        road_data = df[df['road_type_name'] == road_type]
        peak_time = road_data.groupby('time_slot_30min')['passenger_throughput'].mean().idxmax()
        peak_value = road_data.groupby('time_slot_30min')['passenger_throughput'].mean().max()
        print(f"{road_type}: {peak_time} (载人量: {peak_value:.2f})")
    
    # 找出载人量最高的路段
    print("\n载人量最高的前5个路段:")
    top_roads = df.groupby(['road_id', 'road_type_name'])['passenger_throughput'].sum().nlargest(5)
    for (road_id, road_type), throughput in top_roads.items():
        print(f"路段ID: {road_id} ({road_type}) - 载人量: {throughput:.2f}")

def create_detailed_analysis(df):
    """创建详细分析图表"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('载人量详细分析', fontsize=16, fontweight='bold')
    
    # 1. 载人量分布直方图
    axes[0,0].hist(df['passenger_throughput'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_title('载人量分布直方图')
    axes[0,0].set_xlabel('载人量')
    axes[0,0].set_ylabel('频次')
    
    # 2. 各时间段平均载人量
    time_avg = df.groupby('time_slot_30min')['passenger_throughput'].mean().sort_index()
    axes[0,1].bar(range(len(time_avg)), time_avg.values, color='lightcoral')
    axes[0,1].set_title('各时间段平均载人量')
    axes[0,1].set_xlabel('时间段')
    axes[0,1].set_ylabel('平均载人量')
    axes[0,1].set_xticks(range(0, len(time_avg), 4))
    axes[0,1].set_xticklabels(time_avg.index[::4], rotation=45)
    
    # 3. 载人量与车辆数量关系
    axes[0,2].scatter(df['car_count'] + df['ebike_count'] + df['bus_count'], 
                     df['passenger_throughput'], alpha=0.6, color='green')
    axes[0,2].set_title('总车辆数vs载人量')
    axes[0,2].set_xlabel('总车辆数')
    axes[0,2].set_ylabel('载人量')
    
    # 4. 各道路类型在不同时间的载人量变化
    for i, road_type in enumerate(df['road_type_name'].unique()):
        road_data = df[df['road_type_name'] == road_type]
        time_series = road_data.groupby('time_hour')['passenger_throughput'].mean()
        axes[1,0].plot(time_series.index, time_series.values, 
                      marker='o', label=road_type, linewidth=2)
    
    axes[1,0].set_title('24小时载人量变化')
    axes[1,0].set_xlabel('时间（小时）')
    axes[1,0].set_ylabel('平均载人量')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 5. 载人量密度分布
    for road_type in df['road_type_name'].unique():
        road_data = df[df['road_type_name'] == road_type]['passenger_throughput']
        axes[1,1].hist(road_data, alpha=0.5, label=road_type, bins=30, density=True)
    
    axes[1,1].set_title('各道路类型载人量密度分布')
    axes[1,1].set_xlabel('载人量')
    axes[1,1].set_ylabel('密度')
    axes[1,1].legend()
    
    # 6. 载人量与道路长度关系
    if 'road_length_km' in df.columns:
        axes[1,2].scatter(df['road_length_km'], df['passenger_throughput'], 
                         c=df['road_type_code'], cmap='viridis', alpha=0.6)
        axes[1,2].set_title('道路长度vs载人量')
        axes[1,2].set_xlabel('道路长度(km)')
        axes[1,2].set_ylabel('载人量')
        cbar = plt.colorbar(axes[1,2].collections[0], ax=axes[1,2])
        cbar.set_label('道路类型编码')
    else:
        axes[1,2].text(0.5, 0.5, '缺少道路长度数据', ha='center', va='center', 
                      transform=axes[1,2].transAxes, fontsize=12)
        axes[1,2].set_title('道路长度数据缺失')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 加载数据并计算载人量
    df = load_and_calculate_data()
    
    # 进行统计分析
    analyze_peak_hours(df)
    
    # 创建可视化图表
    visualize_passenger_throughput(df)
    
    # 创建详细分析图表
    create_detailed_analysis(df)
    
    print("\n可视化完成！")