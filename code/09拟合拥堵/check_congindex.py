import pandas as pd
import io
import matplotlib.pyplot as plt
import sys
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 获取CSV文件名（支持命令行参数）
if len(sys.argv) > 1:
    csv_path = sys.argv[1]
else:
    csv_path = r"data\10线性规划\03高拥堵.csv"

# 读取CSV文件
df = pd.read_csv(csv_path)

# --- 第一步：根据电动车基础量反推其他交通工具数量 ---
def infer_traffic_counts(row):
    ebike_count = row['ebike_count']
    road_type_code = row['road_type_code']

    # 固定参数 (来自 "优化前基准：电动车基础量反推其他.md")
    bus_ratio = {1: 0.015, 2: 0.04, 3: 0.02, 4: 0.005}
    ebike_ratio = {1: 0.0, 2: 0.06, 3: 0.12, 4: 0.15}
    walk_base_ratio = {1: 0, 2: 0.15, 3: 0.35, 4: 0.60}
    w_walk = 0.15  # 步行人次份额
    O_car_avg = 1.3  # 平均每车人数

    # 求汽车数量
    if ebike_ratio[road_type_code] == 0:
        car_count = 0  # 仅适用于高速公路，ebike_count 应为 0
    else:
        car_count = ebike_count / ebike_ratio[road_type_code]

    # 求公交车数量
    bus_count = car_count * bus_ratio[road_type_code]

    # 求步行者数量
    pedestrian_count = walk_base_ratio[road_type_code] * car_count * O_car_avg * (w_walk / (1 - w_walk))

    return car_count, bus_count, pedestrian_count

df[['car_count', 'bus_count', 'pedestrian_count']] = df.apply(infer_traffic_counts, axis=1, result_type='expand')

# --- 第二步：计算拥堵指数 ---
def calculate_congestion_index(row):
    # 输入参数
    ebike_count = row['ebike_count']
    bus_count = row['bus_count']
    pedestrian_count = row['pedestrian_count']
    car_count = row['car_count']
    road_length_km = row['road_length_km']
    avg_speed_kph = row['avg_speed_kph']
    # road_type_code 不直接用于拥堵指数计算，但用于推导其他变量

    # 固定参数 (来自 "线性规划所用公式.md" - 拥堵指数部分)
    Intercept = 0.1807
    Coef_length_km = 0.0146
    Coef_car_count = 0.0204
    Coef_ebike_count = 0.0319
    Coef_bus_count = 0.0024
    Coef_pedestrian = 0.0204
    Coef_avg_speed = -0.0439

    congestion_index = (
        Intercept
        + Coef_length_km * road_length_km
        + Coef_car_count * car_count
        + Coef_ebike_count * ebike_count
        + Coef_bus_count * bus_count
        + Coef_pedestrian * pedestrian_count
        + Coef_avg_speed * avg_speed_kph
    )
    return congestion_index

df['congestion_index'] = df.apply(calculate_congestion_index, axis=1)

# 显示结果
print(f"计算后的路段数据（文件: {os.path.basename(csv_path)}），包含推导出的交通工具数量和拥堵指数:")
print(df[['road_id', 'road_type_code', 'ebike_count', 'car_count', 'bus_count', 'pedestrian_count', 'road_length_km', 'avg_speed_kph', 'congestion_index']].to_markdown(index=False))

# 绘制拥堵指数分布直方图
plt.figure(figsize=(8, 5))
df['congestion_index'].hist(bins=20)
plt.xlabel('拥堵指数')
plt.ylabel('路段数量')
plt.title(f'拥堵指数分布（{os.path.basename(csv_path)}）')
plt.show()