import pandas as pd
import numpy as np

# 读取CSV文件
input_file = r'data\06二氧化碳.csv'
df = pd.read_csv(input_file)

# 固定参数
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

# 保存结果
output_file = r'E:\github_projects\math_modeling_playground\playground\day4\07安全.csv'
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"已成功计算综合安全指数并保存到: {output_file}")
print(f"数据维度: {df.shape}")
print("\n综合安全指数统计信息:")
print(df['safety_index'].describe())