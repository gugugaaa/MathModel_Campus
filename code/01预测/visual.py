import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
data = pd.read_csv(r'E:\github_projects\math_modeling\data\02重命名.csv')

# 数据预处理
# 将时间转换为小时格式
data['hour'] = pd.to_datetime(data['time_slot_30min'], format='%H:%M').dt.hour
data['minute'] = pd.to_datetime(data['time_slot_30min'], format='%H:%M').dt.minute
data['time_decimal'] = data['hour'] + data['minute'] / 60

# 定义早高峰和晚高峰时间段
def get_peak_period(time_decimal):
    if 7 <= time_decimal <= 9:
        return '早高峰'
    elif 17 <= time_decimal <= 19:
        return '晚高峰'
    else:
        return '其他'

data['peak_period'] = data['time_decimal'].apply(get_peak_period)

# 筛选高峰时段数据
peak_data = data[data['peak_period'].isin(['早高峰', '晚高峰'])]

# 1. 计算平均一条路早高峰、晚高峰的各种车辆数量
print("=== 分析1: 平均一条路早晚高峰各类交通工具数量 ===")
avg_by_peak = peak_data.groupby('peak_period')[['car_count', 'ebike_count', 'bus_count', 'pedestrian_count']].mean()
print(avg_by_peak)

# 可视化1: 早晚高峰平均交通量对比
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
x = np.arange(len(avg_by_peak.columns))
width = 0.35

morning = avg_by_peak.loc['早高峰']
evening = avg_by_peak.loc['晚高峰']

bars1 = ax.bar(x - width/2, morning, width, label='早高峰', alpha=0.8)
bars2 = ax.bar(x + width/2, evening, width, label='晚高峰', alpha=0.8)

ax.set_xlabel('交通工具类型')
ax.set_ylabel('平均数量')
ax.set_title('早晚高峰各类交通工具平均数量对比')
ax.set_xticks(x)
ax.set_xticklabels(['汽车', '电动车', '公交车', '行人'])
ax.legend()

# 在柱状图上添加数值标签
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom')

add_value_labels(bars1)
add_value_labels(bars2)

plt.tight_layout()
plt.savefig(r'E:\github_projects\math_modeling\code\01预测\早晚高峰交通量对比.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. 分析不同道路类型的早晚高峰交通量随时间变化
print("\n=== 分析2: 不同道路类型早晚高峰交通量时间变化 ===")

# 为道路类型添加标签
road_type_labels = {1: '高速', 2: '主干道', 3: '次干道', 4: '支路'}
data['road_type_label'] = data['road_type_code'].map(road_type_labels)

# 计算每个时间段每种道路类型的平均交通量
time_road_analysis = data.groupby(['time_slot_30min', 'road_type_code'])[['car_count', 'ebike_count', 'bus_count', 'pedestrian_count']].mean().reset_index()
time_road_analysis['road_type_label'] = time_road_analysis['road_type_code'].map(road_type_labels)

# 将时间转换为便于绘图的格式
time_road_analysis['time_decimal'] = pd.to_datetime(time_road_analysis['time_slot_30min'], format='%H:%M').dt.hour + pd.to_datetime(time_road_analysis['time_slot_30min'], format='%H:%M').dt.minute / 60

# 创建2x2的子图
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
road_types = [1, 2, 3, 4]
vehicle_types = ['car_count', 'ebike_count', 'bus_count', 'pedestrian_count']
vehicle_labels = ['汽车', '电动车', '公交车', '行人']

for i, road_type in enumerate(road_types):
    row = i // 2
    col = i % 2
    ax = axes[row, col]
    
    # 获取当前道路类型的数据
    road_data = time_road_analysis[time_road_analysis['road_type_code'] == road_type]
    
    # 为每种交通工具绘制曲线
    for j, (vehicle_type, label) in enumerate(zip(vehicle_types, vehicle_labels)):
        ax.plot(road_data['time_decimal'], road_data[vehicle_type], 
                marker='o', linewidth=2, markersize=4, 
                label=label)
    
    # 高亮早晚高峰时段
    ax.axvspan(7, 9, alpha=0.2, color='red', label='早高峰')
    ax.axvspan(17, 19, alpha=0.2, color='orange', label='晚高峰')
    
    ax.set_xlabel('时间 (小时)')
    ax.set_ylabel('交通工具数量')
    ax.set_title(f'{road_type_labels[road_type]}不同出行方式数量随时间变化')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 24)

plt.tight_layout()
plt.savefig(r'E:\github_projects\math_modeling\code\01预测\道路类型交通量时间变化.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. 早晚高峰不同道路类型交通量对比热力图
print("\n=== 分析3: 早晚高峰不同道路类型交通量热力图 ===")

# 筛选早晚高峰数据
peak_road_data = data[data['peak_period'].isin(['早高峰', '晚高峰'])]

# 计算每种道路类型在早晚高峰的平均交通量
heatmap_data = peak_road_data.groupby(['peak_period', 'road_type_code'])[['car_count', 'ebike_count', 'bus_count', 'pedestrian_count']].mean()

# 创建热力图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
vehicle_types = ['car_count', 'ebike_count', 'bus_count', 'pedestrian_count']
vehicle_labels = ['汽车数量', '电动车数量', '公交车数量', '行人数量']

for i, (vehicle_type, label) in enumerate(zip(vehicle_types, vehicle_labels)):
    row = i // 2
    col = i % 2
    ax = axes[row, col]
    
    # 重塑数据为热力图格式
    pivot_data = heatmap_data[vehicle_type].unstack(level=1)
    pivot_data.columns = [road_type_labels[col] for col in pivot_data.columns]
    
    # 绘制热力图
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax, 
                cbar_kws={'label': label})
    ax.set_title(f'{label}热力图')
    ax.set_xlabel('道路类型')
    ax.set_ylabel('高峰时段')

plt.tight_layout()
plt.savefig(r'E:\github_projects\math_modeling\code\01预测\交通量热力图.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. 统计摘要
print("\n=== 统计摘要 ===")
print("各道路类型早晚高峰平均交通量:")
summary = peak_road_data.groupby(['road_type_code', 'peak_period'])[['car_count', 'ebike_count', 'bus_count', 'pedestrian_count']].mean()
for road_type in [1, 2, 3, 4]:
    print(f"\n{road_type_labels[road_type]}:")
    print(summary.loc[road_type])

print("\n数据可视化完成！生成的图表已保存到指定目录。")