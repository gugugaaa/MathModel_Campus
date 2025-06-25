# filepath: e:\github_projects\math_modeling_playground\road_day4\02拥堵\visual.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
data_path = r'data\03拥挤.csv'
df = pd.read_csv(data_path)

# 道路类型映射
road_type_map = {
    1: '高速',
    2: '主干路',
    3: '次干路',
    4: '支路'
}

# 添加道路类型名称
df['road_type_name'] = df['road_type_code'].map(road_type_map)

# 时间处理 - 将时间字符串转换为小时数便于排序
def time_to_hour(time_str):
    hour, minute = map(int, time_str.split(':'))
    return hour + minute/60

df['hour_numeric'] = df['time_slot_30min'].apply(time_to_hour)

# 按道路类型和时间分组，计算平均拥挤指数
grouped_data = df.groupby(['road_type_name', 'time_slot_30min', 'hour_numeric'])['congestion_index'].mean().reset_index()

# 创建图表
plt.figure(figsize=(15, 8))

# 为每种道路类型绘制线条
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
road_types = ['高速', '主干路', '次干路', '支路']

for i, road_type in enumerate(road_types):
    type_data = grouped_data[grouped_data['road_type_name'] == road_type].sort_values('hour_numeric')
    plt.plot(type_data['time_slot_30min'], type_data['congestion_index'], 
             marker='o', linewidth=2.5, markersize=6, 
             color=colors[i], label=road_type, alpha=0.8)

plt.title('各道路类型拥挤程度随时间变化趋势', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('时间段', fontsize=12)
plt.ylabel('拥挤指数', fontsize=12)
plt.legend(fontsize=11, loc='upper right')
plt.grid(True, alpha=0.3)

# 设置x轴标签旋转
plt.xticks(rotation=45, ha='right')

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('road_congestion_by_type_time.png', dpi=300, bbox_inches='tight')
plt.show()

# 输出统计信息
print("各道路类型拥挤指数统计:")
print(df.groupby('road_type_name')['congestion_index'].describe())

# 创建热力图显示更详细的时间-道路类型拥挤模式
plt.figure(figsize=(12, 8))

# 创建透视表
pivot_data = df.groupby(['road_type_name', 'time_slot_30min'])['congestion_index'].mean().unstack(fill_value=0)

# 重新排序时间列
time_order = sorted(pivot_data.columns, key=time_to_hour)
pivot_data = pivot_data[time_order]

# 绘制热力图
sns.heatmap(pivot_data, annot=False, cmap='YlOrRd', cbar_kws={'label': '拥挤指数'})
plt.title('各道路类型拥挤程度热力图', fontsize=16, fontweight='bold')
plt.xlabel('时间段', fontsize=12)
plt.ylabel('道路类型', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# 保存热力图
plt.savefig('road_congestion_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()