import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 读取数据
input_file = r'data\02重命名.csv'
df = pd.read_csv(input_file)

# 固定参数（与create.py保持一致）
k_mm = 1.0      # 机动车–机动车
k_me = 1.5      # 机动车–电动车
k_mp = 2.0      # 机动车–行人
alpha_bus = 2.5 # 公交车等效系数

V0 = 30   # 安全基准速度 km/h
m = 3     # 速度指数

L0 = 1    # km

def calculate_safety_index(row):
    """计算综合安全指数"""
    # 1. 冲突暴露量
    car_eq = row['car_count'] + alpha_bus * row['bus_count']
    exposure_conflict = (
        k_mm * car_eq**2 +
        k_me * car_eq * row['ebike_count'] +
        k_mp * car_eq * row['pedestrian_count']
    )
    
    # 2. 速度风险
    speed_factor = (row['avg_speed_kph'] / V0) ** m
    
    # 3. 长度因子
    length_factor = row['road_length_km'] / L0
    
    # 4. 综合安全指数
    SI_raw = length_factor * exposure_conflict * speed_factor
    SI = 100 / (1 + SI_raw)  # 0–100，越高越安全
    
    return SI

# 计算综合安全指数
df['safety_index'] = df.apply(calculate_safety_index, axis=1)

# 道路类型映射
road_type_map = {
    1: '高速公路',
    2: '主干道',
    3: '次干道',
    4: '支路'
}
df['road_type_name'] = df['road_type_code'].map(road_type_map)

# 创建可视化
plt.figure(figsize=(16, 12))

# 1. 不同道路类型的安全指数箱线图
plt.subplot(2, 3, 1)
sns.boxplot(data=df, x='road_type_name', y='safety_index')
plt.title('不同道路类型的安全指数分布')
plt.xlabel('道路类型')
plt.ylabel('安全指数')
plt.xticks(rotation=45)

# 2. 按时间段的安全指数变化（所有道路类型）
plt.subplot(2, 3, 2)
time_safety = df.groupby('time_slot_30min')['safety_index'].mean().reset_index()
plt.plot(range(len(time_safety)), time_safety['safety_index'], marker='o')
plt.title('全天安全指数变化趋势')
plt.xlabel('时间段（每2小时标记）')
plt.ylabel('平均安全指数')
# 设置x轴标签，每2小时显示一次
step = max(1, len(time_safety) // 12)
plt.xticks(range(0, len(time_safety), step), 
          [time_safety['time_slot_30min'].iloc[i] for i in range(0, len(time_safety), step)], 
          rotation=45)

# 3. 不同道路类型的时间安全指数热力图
plt.subplot(2, 3, 3)
pivot_data = df.groupby(['road_type_name', 'time_slot_30min'])['safety_index'].mean().unstack()
sns.heatmap(pivot_data, cmap='RdYlGn', cbar_kws={'label': '安全指数'})
plt.title('道路类型-时间段安全指数热力图')
plt.xlabel('时间段')
plt.ylabel('道路类型')

# 4. 各道路类型的时间变化趋势
plt.subplot(2, 3, 4)
for road_type in road_type_map.values():
    road_data = df[df['road_type_name'] == road_type]
    time_trend = road_data.groupby('time_slot_30min')['safety_index'].mean()
    plt.plot(range(len(time_trend)), time_trend.values, marker='o', label=road_type, alpha=0.7)

plt.title('各道路类型安全指数时间变化')
plt.xlabel('时间段')
plt.ylabel('平均安全指数')
plt.legend()
plt.xticks(range(0, len(time_trend), step), 
          [time_trend.index[i] for i in range(0, len(time_trend), step)], 
          rotation=45)

# 5. 安全指数统计柱状图
plt.subplot(2, 3, 5)
road_stats = df.groupby('road_type_name')['safety_index'].agg(['mean', 'std']).reset_index()
x = range(len(road_stats))
plt.bar(x, road_stats['mean'], yerr=road_stats['std'], capsize=5, alpha=0.7)
plt.title('各道路类型安全指数均值对比')
plt.xlabel('道路类型')
plt.ylabel('平均安全指数')
plt.xticks(x, road_stats['road_type_name'], rotation=45)

# 6. 安全指数分布直方图
plt.subplot(2, 3, 6)
for road_type in road_type_map.values():
    road_data = df[df['road_type_name'] == road_type]['safety_index']
    plt.hist(road_data, alpha=0.6, label=road_type, bins=20)

plt.title('各道路类型安全指数分布')
plt.xlabel('安全指数')
plt.ylabel('频次')
plt.legend()

plt.tight_layout()
plt.savefig(r'安全指数可视化.png', 
            dpi=300, bbox_inches='tight')
plt.show()

# 输出统计信息
print("=" * 50)
print("安全指数分析报告")
print("=" * 50)
print(f"数据总量: {len(df)} 条记录")
print(f"道路类型数量: {df['road_type_code'].nunique()} 种")
print(f"时间段数量: {df['time_slot_30min'].nunique()} 个")

print("\n各道路类型安全指数统计:")
road_summary = df.groupby('road_type_name')['safety_index'].agg(['count', 'mean', 'std', 'min', 'max'])
print(road_summary.round(3))

print("\n最安全的时间段（前5个）:")
time_safety_sorted = df.groupby('time_slot_30min')['safety_index'].mean().sort_values(ascending=False)
print(time_safety_sorted.head().round(3))

print("\n最不安全的时间段（前5个）:")
print(time_safety_sorted.tail().round(3))

print("\n各道路类型最安全和最不安全的时间段:")
for road_type in road_type_map.values():
    road_data = df[df['road_type_name'] == road_type]
    time_safety = road_data.groupby('time_slot_30min')['safety_index'].mean()
    safest_time = time_safety.idxmax()
    least_safe_time = time_safety.idxmin()
    print(f"{road_type}: 最安全 {safest_time} ({time_safety.max():.2f}), 最不安全 {least_safe_time} ({time_safety.min():.2f})")