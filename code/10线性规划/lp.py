import pandas as pd
import numpy as np

# --- 1. 定义来自您文件的固定参数和模型 ---

# 源文件: 优化前基准：电动车基础量反推其他.md
PARAMS_BASELINE = {
    'bus_ratio': {1: 0.015, 2: 0.04, 3: 0.02, 4: 0.005},
    'ebike_ratio': {1: 0.0, 2: 0.06, 3: 0.12, 4: 0.15},
    'walk_base_ratio': {1: 0, 2: 0.15, 3: 0.35, 4: 0.60},
    'w_walk': 0.15,
    'O_car_avg': 1.3
} #

# 源文件: 线性规划所用公式.md
PARAMS_METRICS = {
    # 载人量
    'P_car_avg': 1.3,
    'P_ebike_avg': 1.1,
    'P_bus_avg': 60,
    # CO2排放
    'EF_car_petrol': 0.131,
    'ratio_petrol': 0.754,
    'EF_car_elec': 0.074,
    'ratio_elec': 0.246,
    'EF_ebike': 0.0066,
    'EF_bus': 0.44,
    # 综合安全指数
    'k_mm': 1.0,
    'k_me': 1.5,
    'k_mp': 2.0,
    'k_ee': 0.6,
    'k_ep': 1.2,
    'alpha_bus': 2.5,
    'V0': 30,
    'm': 3,
    'L0': 1,
    # 拥堵指数
    'Intercept': 0.1807,
    'Coef_length_km': 0.0146,
    'Coef_car_count': 0.0204,
    'Coef_ebike_count': 0.0319,
    'Coef_bus_count': 0.0024,
    'Coef_pedestrian': 0.0204,
    'Coef_avg_speed': -0.0439
} #

# 源文件: 动态优化：电动车该变量转移到其他.md
# 注意：此处使用您提供的百分比范围的平均值
TRANSFER_PROB = {
    '主干道': {
        '高拥堵': {'car': 0.315, 'bus': 0.45, 'walk': 0.15},
        '中拥堵': {'car': 0.28, 'bus': 0.40, 'walk': 0.185},
        '低拥堵': {'car': 0.36, 'bus': 0.30, 'walk': 0.215}
    },
    '次干道': {
        '高拥堵': {'car': 0.26, 'bus': 0.40, 'walk': 0.24},
        '中拥堵': {'car': 0.29, 'bus': 0.35, 'walk': 0.26},
        '低拥堵': {'car': 0.32, 'bus': 0.285, 'walk': 0.285}
    },
    '支路': {
        '高拥堵': {'car': 0.22, 'bus': 0.25, 'walk': 0.40},
        '中拥堵': {'car': 0.24, 'bus': 0.27, 'walk': 0.37},
        '低拥堵': {'car': 0.26, 'bus': 0.30, 'walk': 0.35}
    }
} #

ROAD_TYPE_MAP = {1: '高速公路', 2: '主干道', 3: '次干道', 4: '支路'}


# --- 2. 定义核心计算函数 ---

def calculate_baseline_traffic(ebike_count, road_type_code):
    """根据电动车数量反推其他交通工具数量"""
    ebike_ratio = PARAMS_BASELINE['ebike_ratio'].get(road_type_code)
    if not ebike_ratio or ebike_ratio == 0:
        car_count = 0
    else:
        car_count = ebike_count / ebike_ratio #

    bus_count = car_count * PARAMS_BASELINE['bus_ratio'].get(road_type_code, 0) #
    
    pedestrian_count = (PARAMS_BASELINE['walk_base_ratio'].get(road_type_code, 0) * car_count * PARAMS_BASELINE['O_car_avg'] * (PARAMS_BASELINE['w_walk'] / (1 - PARAMS_BASELINE['w_walk']))) #
    
    return car_count, bus_count, pedestrian_count

def calculate_congestion_index(car_count, ebike_count, bus_count, pedestrian_count, road_length_km, avg_speed_kph):
    """计算拥堵指数"""
    p = PARAMS_METRICS
    congestion_index = (
        p['Intercept'] +
        p['Coef_length_km'] * road_length_km +
        p['Coef_car_count'] * car_count +
        p['Coef_ebike_count'] * ebike_count +
        p['Coef_bus_count'] * bus_count +
        p['Coef_pedestrian'] * pedestrian_count +
        p['Coef_avg_speed'] * avg_speed_kph
    ) #
    return congestion_index

def calculate_co2(car_count, ebike_count, bus_count, road_length_km):
    """计算CO2排放量"""
    p = PARAMS_METRICS
    co2_emission = (
        p['EF_car_petrol'] * p['ratio_petrol'] * car_count +
        p['EF_car_elec'] * p['ratio_elec'] * car_count +
        p['EF_ebike'] * ebike_count +
        p['EF_bus'] * bus_count
    ) * road_length_km #
    return co2_emission

def calculate_safety_index(car_count, ebike_count, bus_count, pedestrian_count, road_length_km, avg_speed_kph):
    """计算综合安全指数"""
    p = PARAMS_METRICS
    car_eq = car_count + p['alpha_bus'] * bus_count #
    
    exposure_conflict = (
        p['k_mm'] * car_eq**2 +
        p['k_me'] * car_eq * ebike_count +
        p['k_mp'] * car_eq * pedestrian_count +
        p['k_ee'] * ebike_count**2 +
        p['k_ep'] * ebike_count * pedestrian_count
    ) #
    
    speed_factor = (avg_speed_kph / p['V0']) ** p['m'] #
    length_factor = road_length_km / p['L0'] #
    
    si_raw = length_factor * exposure_conflict * speed_factor #
    safety_index = 100 / (1 + si_raw) #
    return safety_index

def calculate_passenger_throughput(car_count, ebike_count, bus_count):
    """计算载客量"""
    p = PARAMS_METRICS
    passenger_throughput = (
        p['P_car_avg'] * car_count +
        p['P_ebike_avg'] * ebike_count +
        p['P_bus_avg'] * bus_count
    )
    return passenger_throughput

def solve_optimization(row, congestion_level):
    """对单条道路数据进行优化求解"""
    
    # 1. 提取原始数据
    road_type_code = row['road_type_code']
    original_ebike_count = row['ebike_count']
    road_length_km = row['road_length_km']
    avg_speed_kph = row['avg_speed_kph']
    
    # 2. 计算基准交通流和指标
    car_count_orig, bus_count_orig, ped_count_orig = calculate_baseline_traffic(original_ebike_count, road_type_code)
    
    cong_orig = calculate_congestion_index(car_count_orig, original_ebike_count, bus_count_orig, ped_count_orig, road_length_km, avg_speed_kph)
    co2_orig = calculate_co2(car_count_orig, original_ebike_count, bus_count_orig, road_length_km)
    safety_orig = calculate_safety_index(car_count_orig, original_ebike_count, bus_count_orig, ped_count_orig, road_length_km, avg_speed_kph)
    passenger_orig = calculate_passenger_throughput(car_count_orig, original_ebike_count, bus_count_orig)
    
    # 3. 设定优化目标与决策变量
    # 目标：最小化拥堵指数。根据我们之前的分析，在当前模型下，这意味着将电动车减少到下限。
    new_ebike_count = original_ebike_count * 0.7
    
    # 4. 计算转移后的交通流
    delta_ebike = original_ebike_count - new_ebike_count
    delta_passengers = delta_ebike * PARAMS_METRICS['P_ebike_avg']
    
    road_type_name = ROAD_TYPE_MAP.get(road_type_code)
    if not road_type_name or road_type_name == '高速公路':
        # 如果道路类型不支持或无转移数据，则不进行优化
        return { 'status': 'No solution', **row }

    probs = TRANSFER_PROB[road_type_name][congestion_level]
    # 归一化转移概率，确保转移总和为100%
    prob_sum = sum(probs.values())
    
    # 计算新增的各类交通工具/出行者数量
    delta_car = (delta_passengers * (probs['car']/prob_sum)) / PARAMS_METRICS['P_car_avg']
    delta_bus = (delta_passengers * (probs['bus']/prob_sum)) / PARAMS_METRICS['P_bus_avg']
    delta_ped = delta_passengers * (probs['walk']/prob_sum)
    
    # 新的交通流
    car_count_new = car_count_orig + delta_car
    bus_count_new = bus_count_orig + delta_bus
    ped_count_new = ped_count_orig + delta_ped
    
    # 5. 计算优化后的新指标
    cong_new = calculate_congestion_index(car_count_new, new_ebike_count, bus_count_new, ped_count_new, road_length_km, avg_speed_kph)
    co2_new = calculate_co2(car_count_new, new_ebike_count, bus_count_new, road_length_km)
    safety_new = calculate_safety_index(car_count_new, new_ebike_count, bus_count_new, ped_count_new, road_length_km, avg_speed_kph)
    passenger_new = calculate_passenger_throughput(car_count_new, new_ebike_count, bus_count_new)
    
    # 6. 验证约束条件
    status = "Optimal"
    if not (cong_new < cong_orig and safety_new >= safety_orig * 0.95 and co2_new <= co2_orig * 1.05):
        status = "Constraints not met"

    # 7. 整理并返回结果
    result = {
        'road_id': row['road_id'],
        'status': status,
        'ebike_count_orig': original_ebike_count,
        'optimal_ebike_count': new_ebike_count,
        'cong_index_orig': cong_orig,
        'cong_index_new': cong_new,
        'safety_index_orig': safety_orig,
        'safety_index_new': safety_new,
        'co2_orig': co2_orig,
        'co2_new': co2_new,
        'passenger_orig': passenger_orig,
        'passenger_new': passenger_new,
    }
    return result

# --- 3. 主程序入口 ---

def main(csv_path, congestion_level):
    """
    主函数，读取CSV，处理数据，并输出结果。
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"错误：找不到文件 '{csv_path}'。请检查路径是否正确。")
        return

    # 检查拥堵等级输入是否有效
    valid_levels = ['高拥堵', '中拥堵', '低拥堵']
    if congestion_level not in valid_levels:
        print(f"错误：拥堵等级 '{congestion_level}' 无效。请输入 {valid_levels} 中的一个。")
        return
        
    print(f"正在以“{congestion_level}”情景，对文件 '{csv_path}' 中的数据进行优化...")
    
    # 使用 .apply 方法为每一行数据调用优化函数
    results_list = df.apply(solve_optimization, axis=1, congestion_level=congestion_level)
    
    # 将结果列表转换为DataFrame
    results_df = pd.DataFrame(list(results_list))
    
    print("\n--- 优化结果 ---")
    print(results_df)
    
    # 保存结果到新的CSV文件
    output_filename = 'optimization_results.csv'
    results_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存至 '{output_filename}'")


if __name__ == '__main__':
    # --- 请在这里修改您的文件路径和拥堵等级 ---
    # 1. 指定包含道路数据的主表CSV文件路径
    # 例如: 'C:/Users/YourUser/Documents/traffic_data.csv' 或 'data/my_roads.csv'
    INPUT_CSV_PATH = r'data\10线性规划\02中拥堵.csv'

    # 2. 指定拥堵等级 ('高拥堵', '中拥堵', '低拥堵')
    CONGESTION_SCENARIO = '高拥堵' 
    # -----------------------------------------

    # 运行主程序
    if INPUT_CSV_PATH == 'path/to/your/file.csv':
        print("请在脚本中修改 'INPUT_CSV_PATH' 变量，使其指向您的CSV文件。")
    else:
        main(INPUT_CSV_PATH, CONGESTION_SCENARIO)