import pandas as pd
import numpy as np
import warnings

# 忽略运行时可能出现的除零警告
warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- 1. 定义来自您文件的固定参数和模型 ---

# 源文件: 优化前基准：电动车基础量反推其他.md
PARAMS_BASELINE = {
    'bus_ratio': {1: 0.015, 2: 0.04, 3: 0.02, 4: 0.005},
    'ebike_ratio': {1: 0.0, 2: 0.06, 3: 0.12, 4: 0.15},
    'walk_base_ratio': {1: 0, 2: 0.15, 3: 0.35, 4: 0.60},
    'w_walk': 0.15,
    'O_car_avg': 1.3
}

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
    'k_mm': 1.0, 'k_me': 1.5, 'k_mp': 2.0, 'k_ee': 0.6, 'k_ep': 1.2,
    'alpha_bus': 2.5, 'V0': 30, 'm': 3, 'L0': 1,
    # 拥堵指数
    'Intercept': 0.1807, 'Coef_length_km': 0.0146, 'Coef_car_count': 0.0204,
    'Coef_ebike_count': 0.0319, 'Coef_bus_count': 0.0024,
    'Coef_pedestrian': 0.0204, 'Coef_avg_speed': -0.0439
}

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
}

ROAD_TYPE_MAP = {1: '高速公路', 2: '主干道', 3: '次干道', 4: '支路'}

# --- 2. 新增：定义多目标优化权重和约束容忍度 ---
# 您可以根据策略调整这些权重。权重总和不必为1。
# w_congestion, w_safety, w_co2, w_passengers
WEIGHTS_CONFIG = {
    '高拥堵': {
        '安全优先':   [1.0, 1.5, 0.8, 0.5],
        '效率优先':   [1.5, 0.8, 1.0, 0.5],
        '环保优先':   [0.8, 1.0, 1.5, 0.5],
        '均衡模式':   [1.0, 1.0, 1.0, 1.0],
    },
    '中拥堵': {
        '安全优先':   [0.8, 1.5, 1.0, 0.5],
        '效率优先':   [1.5, 1.0, 0.8, 0.5],
        '环保优先':   [1.0, 0.8, 1.5, 0.5],
        '均衡模式':   [1.0, 1.0, 1.0, 1.0],
    },
    '低拥堵': {
        '安全优先':   [0.5, 1.5, 0.8, 1.0],
        '效率优先':   [1.5, 0.8, 0.5, 1.0],
        '环保优先':   [0.5, 1.0, 1.5, 0.8],
        '均衡模式':   [1.0, 1.0, 1.0, 1.0],
    }
}

# 约束容忍度: 允许新指标在一定程度上“放松”，以寻求更好的综合解
# 例如: safety_new > safety_orig * (1 - tolerance)
# 值为0表示严格遵守，0.05表示允许5%的负向偏差
TOLERANCE_CONFIG = {
    '高拥堵': {'congestion': 0, 'safety': 0.05, 'co2': 0.05, 'passengers': 0.01}, # 高拥堵时，安全和CO2可略微放宽
    '中拥堵': {'congestion': 0, 'safety': 0.02, 'co2': 0.02, 'passengers': 0.01},
    '低拥堵': {'congestion': 0.05, 'safety': 0.01,    'co2': 0.01,    'passengers': 0.01}, # 低拥堵时，要求更严格
}

# --- 3. 定义核心计算函数 (与您版本基本一致) ---

def calculate_baseline_traffic(ebike_count, road_type_code):
    p = PARAMS_BASELINE
    ebike_ratio = p['ebike_ratio'].get(road_type_code, 0)
    car_count = ebike_count / ebike_ratio if ebike_ratio > 0 else 0
    bus_count = car_count * p['bus_ratio'].get(road_type_code, 0)
    pedestrian_count = (p['walk_base_ratio'].get(road_type_code, 0) * car_count * p['O_car_avg'] * (p['w_walk'] / (1 - p['w_walk'])))
    return car_count, bus_count, pedestrian_count

def calculate_all_metrics(car_count, ebike_count, bus_count, pedestrian_count, road_length_km, avg_speed_kph):
    """一个函数计算所有指标"""
    p = PARAMS_METRICS
    # 拥堵指数
    congestion = (
        p['Intercept'] + p['Coef_length_km'] * road_length_km + p['Coef_car_count'] * car_count +
        p['Coef_ebike_count'] * ebike_count + p['Coef_bus_count'] * bus_count +
        p['Coef_pedestrian'] * pedestrian_count + p['Coef_avg_speed'] * avg_speed_kph)
    # CO2
    co2 = (
        p['EF_car_petrol'] * p['ratio_petrol'] * car_count + p['EF_car_elec'] * p['ratio_elec'] * car_count +
        p['EF_ebike'] * ebike_count + p['EF_bus'] * bus_count) * road_length_km
    # 安全指数
    car_eq = car_count + p['alpha_bus'] * bus_count
    exposure = (
        p['k_mm'] * car_eq**2 + p['k_me'] * car_eq * ebike_count + p['k_mp'] * car_eq * pedestrian_count +
        p['k_ee'] * ebike_count**2 + p['k_ep'] * ebike_count * pedestrian_count)
    si_raw = (road_length_km / p['L0']) * exposure * ((avg_speed_kph / p['V0'])**p['m'])
    safety = 100 / (1 + si_raw) if (1 + si_raw) > 0 else 100
    # 载人量
    passengers = (p['P_car_avg'] * car_count + p['P_ebike_avg'] * ebike_count + p['P_bus_avg'] * bus_count)
    
    return {'congestion': congestion, 'co2': co2, 'safety': safety, 'passengers': passengers}

# --- 4. 优化求解核心逻辑 ---

def solve_optimization_for_road(row, congestion_level, weight_scenario):
    """对单条道路数据进行优化求解"""
    
    # 1. 提取原始数据并计算基准指标
    orig_ebike_count = row['ebike_count']
    if orig_ebike_count == 0:
        return {
            'road_id': row['road_id'], 
            'status': 'Skipped (No E-bikes)', 
            'ebike_count_orig': orig_ebike_count,
            'optimal_ebike_count': orig_ebike_count,
            'is_optimized': 0,
            **{f'{k}_orig': 0 for k in ['congestion', 'safety', 'co2', 'passengers']},
            **{f'{k}_new': 0 for k in ['congestion', 'safety', 'co2', 'passengers']}
        }

    road_type_code = row['road_type_code']
    road_length_km = row['road_length_km']
    avg_speed_kph = row['avg_speed_kph']
    road_type_name = ROAD_TYPE_MAP.get(road_type_code)

    if not road_type_name or road_type_name == '高速公路':
        # 计算原始指标用于返回
        car_orig, bus_orig, ped_orig = calculate_baseline_traffic(orig_ebike_count, road_type_code)
        metrics_orig = calculate_all_metrics(car_orig, orig_ebike_count, bus_orig, ped_orig, road_length_km, avg_speed_kph)
        return {
            'road_id': row['road_id'], 
            'status': 'Skipped (Unsupported Road Type)', 
            'ebike_count_orig': orig_ebike_count,
            'optimal_ebike_count': orig_ebike_count,
            'is_optimized': 0,
            **{f'{k}_orig': v for k, v in metrics_orig.items()},
            **{f'{k}_new': v for k, v in metrics_orig.items()}
        }

    car_orig, bus_orig, ped_orig = calculate_baseline_traffic(orig_ebike_count, road_type_code)
    metrics_orig = calculate_all_metrics(car_orig, orig_ebike_count, bus_orig, ped_orig, road_length_km, avg_speed_kph)
    
    # 2. 设定搜索空间和参数
    ebike_min = int(orig_ebike_count * 0.7)
    ebike_max = int(orig_ebike_count * 1.3)
    weights = WEIGHTS_CONFIG[congestion_level][weight_scenario]
    tolerances = TOLERANCE_CONFIG[congestion_level]
    probs = TRANSFER_PROB[road_type_name][congestion_level]
    prob_sum = sum(probs.values()) # 归一化因子
    
    best_solution = None
    min_score = float('inf')

    # 3. 遍历决策变量所有可能的值 (Grid Search)
    # 以步长为1进行搜索，可以调整步长以提高速度
    for new_ebike_count in range(ebike_min, ebike_max + 1):
        # 4. 计算转移后的交通流
        delta_ebike = orig_ebike_count - new_ebike_count
        delta_passengers = delta_ebike * PARAMS_METRICS['P_ebike_avg']
        
        delta_car = (delta_passengers * (probs['car']/prob_sum)) / PARAMS_METRICS['P_car_avg']
        delta_bus = (delta_passengers * (probs['bus']/prob_sum)) / PARAMS_METRICS['P_bus_avg']
        delta_ped = delta_passengers * (probs['walk']/prob_sum)
        
        car_new = car_orig + delta_car
        bus_new = bus_orig + delta_bus
        ped_new = ped_orig + delta_ped
        
        # 5. 计算新方案的各项指标
        metrics_new = calculate_all_metrics(car_new, new_ebike_count, bus_new, ped_new, road_length_km, avg_speed_kph)

        # 6. 验证约束条件 (含容忍度)
        constraints_met = (
            metrics_new['congestion'] < metrics_orig['congestion'] * (1 + tolerances['congestion']) and
            metrics_new['safety']     > metrics_orig['safety']     * (1 - tolerances['safety']) and
            metrics_new['co2']        < metrics_orig['co2']        * (1 + tolerances['co2']) and
            metrics_new['passengers'] > metrics_orig['passengers'] * (1 - tolerances['passengers'])
        )
        
        if constraints_met:
            # 7. 如果满足约束，计算加权目标函数得分
            # 归一化处理（此处使用相对变化作为简化的归一化方法）
            # (new - old) / old
            norm_cong = (metrics_new['congestion'] - metrics_orig['congestion']) / metrics_orig['congestion']
            norm_safe = (metrics_new['safety'] - metrics_orig['safety']) / metrics_orig['safety']
            norm_co2 = (metrics_new['co2'] - metrics_orig['co2']) / metrics_orig['co2']
            norm_pass = (metrics_new['passengers'] - metrics_orig['passengers']) / metrics_orig['passengers']
            
            # 加权求和，我们的目标是最小化这个分数
            score = (weights[0] * norm_cong - 
                     weights[1] * norm_safe + 
                     weights[2] * norm_co2  -
                     weights[3] * norm_pass)

            if score < min_score:
                min_score = score
                best_solution = {
                    'status': 'Optimal',
                    'optimal_ebike_count': new_ebike_count,
                    'is_optimized': 1,
                    **{f'{k}_new': v for k, v in metrics_new.items()}
                }

    # 8. 整理并返回结果
    if best_solution:
        result = {
            'road_id': row['road_id'],
            'ebike_count_orig': orig_ebike_count,
            **{f'{k}_orig': v for k, v in metrics_orig.items()},
            **best_solution
        }
    else:
        # 无法优化时，使用原始值
        result = {
            'road_id': row['road_id'],
            'status': 'No optimization (Constraints not met)',
            'ebike_count_orig': orig_ebike_count,
            'optimal_ebike_count': orig_ebike_count,
            'is_optimized': 0,
            **{f'{k}_orig': v for k, v in metrics_orig.items()},
            **{f'{k}_new': v for k, v in metrics_orig.items()}
        }
    return result

# --- 5. 主程序入口 ---

def main(csv_path, congestion_level, weight_scenario):
    """主函数，读取CSV，分块处理数据，并输出结果。"""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"错误：找不到文件 '{csv_path}'。请检查路径是否正确。")
        return

    valid_levels = ['高拥堵', '中拥堵', '低拥堵']
    if congestion_level not in valid_levels:
        print(f"错误：拥堵等级 '{congestion_level}' 无效。请输入 {valid_levels} 中的一个。")
        return
    
    valid_scenarios = WEIGHTS_CONFIG[congestion_level].keys()
    if weight_scenario not in valid_scenarios:
        print(f"错误：权重情景 '{weight_scenario}' 无效。对于'{congestion_level}', 请输入 {list(valid_scenarios)} 中的一个。")
        return
        
    print(f"开始优化... 文件: '{csv_path}'")
    print(f"拥堵情景: '{congestion_level}', 优化策略: '{weight_scenario}'")
    
    # 建议的分块处理 (chunksize=100)
    chunk_size = 100
    results_list = []
    
    # 将DataFrame分割成块
    list_df = [df[i:i+chunk_size] for i in range(0, df.shape[0], chunk_size)]
    
    for i, chunk in enumerate(list_df):
        print(f"正在处理第 {i+1}/{len(list_df)} 块数据...")
        # 对每个块应用优化函数
        chunk_results = chunk.apply(solve_optimization_for_road, axis=1, 
                                    congestion_level=congestion_level, 
                                    weight_scenario=weight_scenario)
        results_list.extend(list(chunk_results))

    results_df = pd.DataFrame(results_list)
    
    # 统计优化情况
    optimized_count = results_df['is_optimized'].sum()
    total_count = len(results_df)
    optimization_rate = optimized_count / total_count * 100 if total_count > 0 else 0
    
    print("\n--- 优化结果摘要 ---")
    print(f"总道路数量: {total_count}")
    print(f"成功优化数量: {optimized_count}")
    print(f"优化率: {optimization_rate:.2f}%")
    print("\n详细结果预览:")
    print(results_df[['road_id', 'status', 'is_optimized', 'ebike_count_orig', 'optimal_ebike_count', 'congestion_orig', 'congestion_new', 'safety_orig', 'safety_new']].round(2))
    
    output_filename = f'优化_{congestion_level}_{weight_scenario}.csv'
    results_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存至 '{output_filename}'")


if __name__ == '__main__':
    # --- 请在这里修改您的文件路径、拥堵等级和优化策略 ---
    
    # 1. 指定包含道路数据的主表CSV文件路径
    # 例如: 'data/high_congestion.csv'
    INPUT_CSV_PATH = r'data\10线性规划\03高拥堵.csv'

    # 2. 指定拥堵等级 ('高拥堵', '中拥堵', '低拥堵')
    CONGESTION_SCENARIO = '高拥堵' 

    # 3. 指定优化权重策略 ('安全优先', '效率优先', '环保优先', '均衡模式')
    WEIGHT_SCENARIO = '均衡模式'
    
    # ----------------------------------------------------

    if 'path/to/your/file.csv' in INPUT_CSV_PATH:
        print("请在脚本中修改 'INPUT_CSV_PATH' 变量，使其指向您的CSV文件。")
    else:
        main(INPUT_CSV_PATH, CONGESTION_SCENARIO, WEIGHT_SCENARIO)