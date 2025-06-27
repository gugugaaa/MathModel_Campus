import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
from matplotlib.patches import Rectangle

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 忽略运行时可能出现的除零警告
warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- 复制原代码中的参数定义 ---
PARAMS_BASELINE = {
    'bus_ratio': {1: 0.015, 2: 0.04, 3: 0.02, 4: 0.005},
    'ebike_ratio': {1: 0.0, 2: 0.06, 3: 0.12, 4: 0.15},
    'walk_base_ratio': {1: 0, 2: 0.15, 3: 0.35, 4: 0.60},
    'w_walk': 0.15,
    'O_car_avg': 1.3
}

PARAMS_METRICS = {
    'P_car_avg': 1.3,
    'P_ebike_avg': 1.1,
    'P_bus_avg': 60,
    'EF_car_petrol': 0.131,
    'ratio_petrol': 0.754,
    'EF_car_elec': 0.074,
    'ratio_elec': 0.246,
    'EF_ebike': 0.002,
    'EF_bus': 0.44,
    'EF_shared_bike': 0,
    'k_mm': 1.0, 'k_me': 1.5, 'k_mp': 2.0, 'k_ee': 0.6, 'k_ep': 1.2,
    'alpha_bus': 2.5, 'V0': 30, 'm': 3, 'L0': 1,
    'Intercept': 0.1807, 'Coef_length_km': 0.0146, 'Coef_car_count': 0.0204,
    'Coef_ebike_count': 0.0319,
    'Coef_bus_count': 0.0024,
    'Coef_pedestrian': 0.0204, 'Coef_avg_speed': -0.0439
}

TRANSFER_PROB_WITH_SHARED_BIKE = {
    '主干道': {
        '高拥堵': {'car': 0.30, 'bus': 0.40, 'walk': 0.15, 'shared_bike': 0.15},
        '中拥堵': {'car': 0.30, 'bus': 0.35, 'walk': 0.15, 'shared_bike': 0.20},
        '低拥堵': {'car': 0.35, 'bus': 0.35, 'walk': 0.15, 'shared_bike': 0.15}
    },
    '次干道': {
        '高拥堵': {'car': 0.22, 'bus': 0.38, 'walk': 0.22, 'shared_bike': 0.18},
        '中拥堵': {'car': 0.25, 'bus': 0.35, 'walk': 0.22, 'shared_bike': 0.18},
        '低拥堵': {'car': 0.30, 'bus': 0.30, 'walk': 0.20, 'shared_bike': 0.20}
    },
    '支路': {
        '高拥堵': {'car': 0.10, 'bus': 0.20, 'walk': 0.35, 'shared_bike': 0.35},
        '中拥堵': {'car': 0.13, 'bus': 0.20, 'walk': 0.30, 'shared_bike': 0.37},
        '低拥堵': {'car': 0.18, 'bus': 0.15, 'walk': 0.30, 'shared_bike': 0.37}
    }
}

WEIGHTS_CONFIG = {
    '高拥堵': {
        '安全优先': [1.0, 1.5, 0.8, 0.5],
        '效率优先': [1.5, 0.8, 1.0, 0.5],
        '环保优先': [0.8, 1.0, 1.5, 0.5],
        '均衡模式': [1.0, 1.0, 1.0, 1.0],
    },
    '中拥堵': {
        '安全优先': [0.8, 1.5, 1.0, 0.5],
        '效率优先': [1.5, 1.0, 0.8, 0.5],
        '环保优先': [1.0, 0.8, 1.5, 0.5],
        '均衡模式': [1.0, 1.0, 1.0, 1.0],
    },
    '低拥堵': {
        '安全优先': [0.5, 1.5, 0.8, 1.0],
        '效率优先': [1.5, 0.8, 0.5, 1.0],
        '环保优先': [0.5, 1.0, 1.5, 0.8],
        '均衡模式': [1.0, 1.0, 1.0, 1.0],
    }
}

TOLERANCE_CONFIG = {
    '高拥堵': {'congestion': 0, 'safety': 0.05, 'co2': 0.05, 'passengers': 0.005},
    '中拥堵': {'congestion': 0, 'safety': 0.02, 'co2': 0.02, 'passengers': 0.01},
    '低拥堵': {'congestion': 0.05, 'safety': 0.01, 'co2': 0.01, 'passengers': 0.01},
}

ROAD_TYPE_MAP = {1: '高速公路', 2: '主干道', 3: '次干道', 4: '支路'}

# --- 复制原代码中的计算函数 ---
def calculate_baseline_traffic(ebike_count, road_type_code):
    p = PARAMS_BASELINE
    ebike_ratio = p['ebike_ratio'].get(road_type_code, 0)
    car_count = ebike_count / ebike_ratio if ebike_ratio > 0 else 0
    bus_count = car_count * p['bus_ratio'].get(road_type_code, 0)
    pedestrian_count = (p['walk_base_ratio'].get(road_type_code, 0) * car_count * p['O_car_avg'] * (p['w_walk'] / (1 - p['w_walk'])))
    return car_count, bus_count, pedestrian_count

def calculate_all_metrics(car_count, ebike_count, bus_count, pedestrian_count, shared_bike_count, road_length_km, avg_speed_kph):
    p = PARAMS_METRICS
    
    congestion = (
        p['Intercept'] + p['Coef_length_km'] * road_length_km + p['Coef_car_count'] * car_count +
        p['Coef_ebike_count'] * (ebike_count + shared_bike_count) + 
        p['Coef_bus_count'] * bus_count + p['Coef_pedestrian'] * pedestrian_count + 
        p['Coef_avg_speed'] * avg_speed_kph)
    
    co2 = (
        p['EF_car_petrol'] * p['ratio_petrol'] * car_count + 
        p['EF_car_elec'] * p['ratio_elec'] * car_count +
        p['EF_ebike'] * ebike_count + 
        p['EF_bus'] * bus_count +
        p['EF_shared_bike'] * shared_bike_count) * road_length_km

    non_motor_count = ebike_count + shared_bike_count
    car_eq = car_count + p['alpha_bus'] * bus_count
    exposure = (
        p['k_mm'] * car_eq**2 +
        p['k_me'] * car_eq * non_motor_count +
        p['k_mp'] * car_eq * pedestrian_count +
        p['k_ee'] * non_motor_count**2 +
        p['k_ep'] * non_motor_count * pedestrian_count
    )
    si_raw = (road_length_km / p['L0']) * exposure * ((avg_speed_kph / p['V0'])**p['m'])
    safety = 100 / (1 + si_raw) if (1 + si_raw) > 0 else 100
    
    passengers = (
        p['P_car_avg'] * car_count + 
        p['P_ebike_avg'] * ebike_count + 
        p['P_bus_avg'] * bus_count +
        p['P_ebike_avg'] * shared_bike_count + 
        pedestrian_count
    )
    
    return {'congestion': congestion, 'co2': co2, 'safety': safety, 'passengers': passengers}

def visualize_optimization_process(csv_path, congestion_level='低拥堵', weight_scenario='均衡模式', row_index=0):
    """
    可视化第一条数据的优化拟合过程
    
    Parameters:
    csv_path: CSV文件路径
    congestion_level: 拥堵等级
    weight_scenario: 权重策略
    row_index: 要可视化的数据行索引（默认0为第一条）
    """
    
    # 读取数据
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"错误：找不到文件 '{csv_path}'")
        return
    
    if row_index >= len(df):
        print(f"错误：行索引 {row_index} 超出数据范围")
        return
    
    # 获取第一条数据
    row = df.iloc[row_index]
    
    # 提取参数
    orig_ebike_count = row['ebike_count']
    road_type_code = row['road_type_code']
    road_length_km = row['road_length_km']
    avg_speed_kph = row['avg_speed_kph']
    road_type_name = ROAD_TYPE_MAP.get(road_type_code)
    
    if orig_ebike_count == 0 or not road_type_name or road_type_name == '高速公路':
        print("该道路数据无法进行优化")
        return
    
    # 计算基准指标
    car_orig, bus_orig, ped_orig = calculate_baseline_traffic(orig_ebike_count, road_type_code)
    metrics_orig = calculate_all_metrics(car_orig, orig_ebike_count, bus_orig, ped_orig, 0, road_length_km, avg_speed_kph)
    
    # 设定搜索范围
    ebike_min = int(orig_ebike_count * 0.7)
    ebike_max = int(orig_ebike_count * 1.3)
    weights = WEIGHTS_CONFIG[congestion_level][weight_scenario]
    tolerances = TOLERANCE_CONFIG[congestion_level]
    probs = TRANSFER_PROB_WITH_SHARED_BIKE[road_type_name][congestion_level]
    prob_sum = sum(probs.values())
    
    # 存储所有方案的数据
    ebike_range = list(range(ebike_min, ebike_max + 1))
    results = {
        'ebike_count': [],
        'congestion': [],
        'safety': [],
        'co2': [],
        'passengers': [],
        'shared_bike': [],
        'score': [],
        'feasible': []
    }
    
    best_solution = None
    min_score = float('inf')
    
    # 遍历所有可能的电动车数量
    for new_ebike_count in ebike_range:
        # 计算转移后的交通流
        delta_ebike = orig_ebike_count - new_ebike_count
        delta_passengers = delta_ebike * PARAMS_METRICS['P_ebike_avg']
        
        delta_car = (delta_passengers * (probs['car']/prob_sum)) / PARAMS_METRICS['P_car_avg']
        delta_bus = (delta_passengers * (probs['bus']/prob_sum)) / PARAMS_METRICS['P_bus_avg']
        delta_ped = delta_passengers * (probs['walk']/prob_sum)
        delta_shared_bike = (delta_passengers * (probs['shared_bike']/prob_sum)) / PARAMS_METRICS['P_ebike_avg']
        
        car_new = car_orig + delta_car
        bus_new = bus_orig + delta_bus
        ped_new = ped_orig + delta_ped
        shared_bike_new = delta_shared_bike
        
        # 计算新方案的各项指标
        metrics_new = calculate_all_metrics(car_new, new_ebike_count, bus_new, ped_new, shared_bike_new, road_length_km, avg_speed_kph)
        
        # 验证约束条件
        constraints_met = (
            metrics_new['congestion'] < metrics_orig['congestion'] * (1 + tolerances['congestion']) and
            metrics_new['safety'] > metrics_orig['safety'] * (1 - tolerances['safety']) and
            metrics_new['co2'] < metrics_orig['co2'] * (1 + tolerances['co2']) and
            metrics_new['passengers'] > metrics_orig['passengers'] * (1 - tolerances['passengers'])
        )
        
        # 计算目标函数得分
        norm_cong = (metrics_new['congestion'] - metrics_orig['congestion']) / (metrics_orig['congestion'] + 1e-9)
        norm_safe = (metrics_new['safety'] - metrics_orig['safety']) / (metrics_orig['safety'] + 1e-9)
        norm_co2 = (metrics_new['co2'] - metrics_orig['co2']) / (metrics_orig['co2'] + 1e-9)
        norm_pass = (metrics_new['passengers'] - metrics_orig['passengers']) / (metrics_orig['passengers'] + 1e-9)
        
        score = (weights[0] * norm_cong - weights[1] * norm_safe + weights[2] * norm_co2 - weights[3] * norm_pass)
        
        # 存储结果
        results['ebike_count'].append(new_ebike_count)
        results['congestion'].append(metrics_new['congestion'])
        results['safety'].append(metrics_new['safety'])
        results['co2'].append(metrics_new['co2'])
        results['passengers'].append(metrics_new['passengers'])
        results['shared_bike'].append(shared_bike_new)
        results['score'].append(score)
        results['feasible'].append(constraints_met)
        
        # 更新最优解
        if constraints_met and score < min_score:
            min_score = score
            best_solution = {
                'ebike_count': new_ebike_count,
                'shared_bike': shared_bike_new,
                'metrics': metrics_new
            }
    
    # 创建可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'道路优化过程可视化 (道路ID: {row["road_id"]}, {road_type_name}, {congestion_level}, {weight_scenario})', 
                 fontsize=16, fontweight='bold')
    
    # 定义颜色
    feasible_color = 'lightblue'
    infeasible_color = 'lightcoral'
    optimal_color = 'red'
    baseline_color = 'green'
    
    # 1. 拥堵指数
    ax1 = axes[0, 0]
    colors = [feasible_color if f else infeasible_color for f in results['feasible']]
    ax1.scatter(results['ebike_count'], results['congestion'], c=colors, alpha=0.7, s=30)
    ax1.axhline(y=metrics_orig['congestion'], color=baseline_color, linestyle='--', label='基准值')
    ax1.axhline(y=metrics_orig['congestion'] * (1 + tolerances['congestion']), 
                color='orange', linestyle=':', label='约束上限')
    if best_solution:
        ax1.scatter(best_solution['ebike_count'], best_solution['metrics']['congestion'], 
                   color=optimal_color, s=100, marker='*', label='最优解')
    ax1.set_xlabel('电动车数量')
    ax1.set_ylabel('拥堵指数')
    ax1.set_title('拥堵指数变化')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 安全指数
    ax2 = axes[0, 1]
    ax2.scatter(results['ebike_count'], results['safety'], c=colors, alpha=0.7, s=30)
    ax2.axhline(y=metrics_orig['safety'], color=baseline_color, linestyle='--', label='基准值')
    ax2.axhline(y=metrics_orig['safety'] * (1 - tolerances['safety']), 
                color='orange', linestyle=':', label='约束下限')
    if best_solution:
        ax2.scatter(best_solution['ebike_count'], best_solution['metrics']['safety'], 
                   color=optimal_color, s=100, marker='*', label='最优解')
    ax2.set_xlabel('电动车数量')
    ax2.set_ylabel('安全指数')
    ax2.set_title('安全指数变化')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. CO2排放
    ax3 = axes[0, 2]
    ax3.scatter(results['ebike_count'], results['co2'], c=colors, alpha=0.7, s=30)
    ax3.axhline(y=metrics_orig['co2'], color=baseline_color, linestyle='--', label='基准值')
    ax3.axhline(y=metrics_orig['co2'] * (1 + tolerances['co2']), 
                color='orange', linestyle=':', label='约束上限')
    if best_solution:
        ax3.scatter(best_solution['ebike_count'], best_solution['metrics']['co2'], 
                   color=optimal_color, s=100, marker='*', label='最优解')
    ax3.set_xlabel('电动车数量')
    ax3.set_ylabel('CO2排放')
    ax3.set_title('CO2排放变化')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 载客量
    ax4 = axes[1, 0]
    ax4.scatter(results['ebike_count'], results['passengers'], c=colors, alpha=0.7, s=30)
    ax4.axhline(y=metrics_orig['passengers'], color=baseline_color, linestyle='--', label='基准值')
    ax4.axhline(y=metrics_orig['passengers'] * (1 - tolerances['passengers']), 
                color='orange', linestyle=':', label='约束下限')
    if best_solution:
        ax4.scatter(best_solution['ebike_count'], best_solution['metrics']['passengers'], 
                   color=optimal_color, s=100, marker='*', label='最优解')
    ax4.set_xlabel('电动车数量')
    ax4.set_ylabel('总出行人数')
    ax4.set_title('总出行人数变化')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 共享单车数量
    ax5 = axes[1, 1]
    ax5.scatter(results['ebike_count'], results['shared_bike'], c=colors, alpha=0.7, s=30)
    if best_solution:
        ax5.scatter(best_solution['ebike_count'], best_solution['shared_bike'], 
                   color=optimal_color, s=100, marker='*', label='最优解')
    ax5.set_xlabel('电动车数量')
    ax5.set_ylabel('共享单车数量')
    ax5.set_title('共享单车配置')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 目标函数得分
    ax6 = axes[1, 2]
    ax6.scatter(results['ebike_count'], results['score'], c=colors, alpha=0.7, s=30)
    if best_solution:
        ax6.scatter(best_solution['ebike_count'], min_score, 
                   color=optimal_color, s=100, marker='*', label='最优解')
    ax6.set_xlabel('电动车数量')
    ax6.set_ylabel('目标函数得分')
    ax6.set_title('目标函数得分（越小越好）')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 添加图例说明
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor=feasible_color, label='可行解'),
        plt.Rectangle((0,0),1,1, facecolor=infeasible_color, label='不可行解'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=optimal_color, 
                  markersize=10, label='最优解'),
        plt.Line2D([0], [0], color=baseline_color, linestyle='--', label='基准值'),
        plt.Line2D([0], [0], color='orange', linestyle=':', label='约束边界')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # 输出优化结果摘要
    print("\n=== 优化结果摘要 ===")
    print(f"道路ID: {row['road_id']}")
    print(f"道路类型: {road_type_name}")
    print(f"道路长度: {road_length_km:.2f} km")
    print(f"平均速度: {avg_speed_kph:.2f} km/h")
    print(f"拥堵场景: {congestion_level}")
    print(f"优化策略: {weight_scenario}")
    print(f"\n原始电动车数量: {orig_ebike_count}")
    
    if best_solution:
        print(f"最优电动车数量: {best_solution['ebike_count']}")
        print(f"最优共享单车数量: {best_solution['shared_bike']:.2f}")
        print(f"\n指标改善情况:")
        print(f"  拥堵指数: {metrics_orig['congestion']:.4f} → {best_solution['metrics']['congestion']:.4f}")
        print(f"  安全指数: {metrics_orig['safety']:.4f} → {best_solution['metrics']['safety']:.4f}")
        print(f"  CO2排放: {metrics_orig['co2']:.4f} → {best_solution['metrics']['co2']:.4f}")
        print(f"  总出行人数: {metrics_orig['passengers']:.4f} → {best_solution['metrics']['passengers']:.4f}")
    else:
        print("未找到满足约束条件的最优解")
    
    plt.show()

# 使用示例
if __name__ == '__main__':
    # 请修改为您的文件路径
    csv_path = r'data\10线性规划\03高拥堵.csv'
    
    # 可视化第一条数据的优化过程
    visualize_optimization_process(
        csv_path=csv_path,
        congestion_level='高拥堵',
        weight_scenario='均衡模式',
        row_index=0  # 第一条数据
    )