# filepath: e:\github_projects\math_modeling_playground\road_day4\06二氧化碳\visual.py

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

# 二氧化碳排放计算的固定参数
EF_car_petrol = 0.131  # kg CO₂ / 车·km（汽油车）
ratio_petrol  = 0.754  # 小汽车中汽油车占比
EF_car_elec   = 0.074  # kg CO₂ / 车·km（电动小汽车）
ratio_elec    = 0.246  # 小汽车中电动占比
EF_ebike      = 0.0066 # kg CO₂ / 车·km（电动自行车）
EF_bus        = 0.44   # kg CO₂ / 车·km（深圳公交为纯电）

def load_and_calculate_co2():
    """
    加载数据并计算二氧化碳排放量
    """
    # 读取CSV文件
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(input_file, encoding='gbk')
    
    print(f"原始数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    
    # 计算二氧化碳排放量
    df['co2_emission'] = (
        EF_car_petrol * ratio_petrol * df['car_count'] +
        EF_car_elec   * ratio_elec   * df['car_count'] +
        EF_ebike * df['ebike_count'] +
        EF_bus   * df['bus_count']
    ) * df['road_length_km']
    
    # 添加道路类型名称
    road_type_names = {1: '高速公路', 2: '主干道', 3: '次干道', 4: '支路'}
    df['road_type_name'] = df['road_type_code'].map(road_type_names)
    
    # 转换时间格式用于排序
    df['time_hour'] = df['time_slot_30min'].apply(lambda x: int(x.split(':')[0]) + int(x.split(':')[1])/60)
    
    return df

def visualize_co2_by_road_type_and_time(df):
    """
    按道路类型和时间段可视化二氧化碳排放量
    """
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('不同道路类型在各时间段的二氧化碳排放量分析', fontsize=16, fontweight='bold')
    
    # 1. 按时间段和道路类型的总排放量
    ax1 = axes[0, 0]
    time_road_co2 = df.groupby(['time_slot_30min', 'road_type_name'])['co2_emission'].sum().unstack(fill_value=0)
    time_road_co2.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('各时间段不同道路类型二氧化碳排放量')
    ax1.set_xlabel('时间段')
    ax1.set_ylabel('二氧化碳排放量 (kg)')
    ax1.legend(title='道路类型', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. 热力图显示道路类型与时间段的关系
    ax2 = axes[0, 1]
    pivot_data = df.pivot_table(values='co2_emission', index='road_type_name', 
                               columns='time_slot_30min', aggfunc='sum', fill_value=0)
    sns.heatmap(pivot_data, annot=False, fmt='.0f', cmap='YlOrRd', ax=ax2)
    ax2.set_title('道路类型-时间段二氧化碳排放热力图')
    ax2.set_xlabel('时间段')
    ax2.set_ylabel('道路类型')
    
    # 3. 各道路类型的总排放量占比
    ax3 = axes[1, 0]
    road_total_co2 = df.groupby('road_type_name')['co2_emission'].sum()
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    wedges, texts, autotexts = ax3.pie(road_total_co2.values, labels=road_total_co2.index, 
                                       autopct='%1.1f%%', colors=colors, startangle=90)
    ax3.set_title('各道路类型二氧化碳排放量占比')
    
    # 4. 时间趋势线图
    ax4 = axes[1, 1]
    for road_type in df['road_type_name'].unique():
        road_data = df[df['road_type_name'] == road_type]
        time_co2 = road_data.groupby('time_hour')['co2_emission'].mean()
        ax4.plot(time_co2.index, time_co2.values, marker='o', label=road_type, linewidth=2)
    
    ax4.set_title('各道路类型二氧化碳排放量时间趋势')
    ax4.set_xlabel('时间 (小时)')
    ax4.set_ylabel('平均二氧化碳排放量 (kg)')
    ax4.legend(title='道路类型')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def detailed_analysis(df):
    """
    详细统计分析
    """
    print("\n=== 二氧化碳排放量详细分析 ===")
    
    # 总体统计
    print(f"\n总体统计:")
    print(f"总二氧化碳排放量: {df['co2_emission'].sum():.2f} kg")
    print(f"平均排放量: {df['co2_emission'].mean():.4f} kg")
    print(f"最大排放量: {df['co2_emission'].max():.4f} kg")
    print(f"最小排放量: {df['co2_emission'].min():.4f} kg")
    
    # 按道路类型统计
    print(f"\n按道路类型统计:")
    road_stats = df.groupby('road_type_name')['co2_emission'].agg(['sum', 'mean', 'std', 'count'])
    road_stats.columns = ['总排放量(kg)', '平均排放量(kg)', '标准差', '记录数']
    print(road_stats.round(4))
    
    # 按时间段统计
    print(f"\n按时间段统计:")
    time_stats = df.groupby('time_slot_30min')['co2_emission'].agg(['sum', 'mean'])
    time_stats.columns = ['总排放量(kg)', '平均排放量(kg)']
    print(time_stats.round(4))
    
    # 找出排放量最高的路段和时间
    max_emission_idx = df['co2_emission'].idxmax()
    max_emission_row = df.loc[max_emission_idx]
    print(f"\n排放量最高的记录:")
    print(f"路段ID: {max_emission_row['road_id']}")
    print(f"道路类型: {max_emission_row['road_type_name']}")
    print(f"时间段: {max_emission_row['time_slot_30min']}")
    print(f"排放量: {max_emission_row['co2_emission']:.4f} kg")

def create_additional_visualizations(df):
    """
    创建额外的可视化图表
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('二氧化碳排放量补充分析', fontsize=16, fontweight='bold')
    
    # 1. 各车辆类型对排放量的贡献
    ax1 = axes[0, 0]
    
    # 计算各车辆类型的排放贡献
    df['car_emission'] = (EF_car_petrol * ratio_petrol + EF_car_elec * ratio_elec) * df['car_count'] * df['road_length_km']
    df['ebike_emission'] = EF_ebike * df['ebike_count'] * df['road_length_km']
    df['bus_emission'] = EF_bus * df['bus_count'] * df['road_length_km']
    
    vehicle_emissions = {
        '小汽车': df['car_emission'].sum(),
        '电动自行车': df['ebike_emission'].sum(),
        '公交车': df['bus_emission'].sum()
    }
    
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    ax1.pie(vehicle_emissions.values(), labels=vehicle_emissions.keys(), 
            autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('各车辆类型二氧化碳排放贡献')
    
    # 2. 道路长度与排放量的关系
    ax2 = axes[0, 1]
    scatter = ax2.scatter(df['road_length_km'], df['co2_emission'], 
                         c=df['road_type_code'], cmap='viridis', alpha=0.6)
    ax2.set_xlabel('道路长度 (km)')
    ax2.set_ylabel('二氧化碳排放量 (kg)')
    ax2.set_title('道路长度与二氧化碳排放量关系')
    plt.colorbar(scatter, ax=ax2, label='道路类型编码')
    
    # 3. 交通流量与排放量的关系
    ax3 = axes[1, 0]
    ax3.scatter(df['traffic_flow_vph'], df['co2_emission'], alpha=0.6, color='orange')
    ax3.set_xlabel('交通流量 (辆/h)')
    ax3.set_ylabel('二氧化碳排放量 (kg)')
    ax3.set_title('交通流量与二氧化碳排放量关系')
    
    # 4. 各道路类型的箱线图
    ax4 = axes[1, 1]
    df.boxplot(column='co2_emission', by='road_type_name', ax=ax4)
    ax4.set_title('各道路类型二氧化碳排放量分布')
    ax4.set_xlabel('道路类型')
    ax4.set_ylabel('二氧化碳排放量 (kg)')
    
    plt.tight_layout()
    plt.show()

def main():
    """
    主函数
    """
    print("开始分析二氧化碳排放量...")
    
    # 加载数据并计算排放量
    df = load_and_calculate_co2()
    
    # 详细统计分析
    detailed_analysis(df)
    
    # 主要可视化
    visualize_co2_by_road_type_and_time(df)
    
    # 补充可视化
    create_additional_visualizations(df)
    
    print("\n分析完成！")

if __name__ == "__main__":
    main()