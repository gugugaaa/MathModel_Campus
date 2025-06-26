import pandas as pd
import numpy as np
import os

def calculate_congestion_index():
    """
    基于改进版公式计算拥堵指数
    使用PCU权重来更真实地反映不同交通工具对拥堵的贡献
    """
    
    # 1. 固定参数设置
    # PCU权重 (各类交通工具的拥堵贡献权重)
    pcu_weights = {
        'ebike': 0.4,   # 电动车：体积小但穿插频繁
        'car': 1.0,     # 小汽车：作为标准基准
        'bus': 2.5      # 公交车：体积大，启停影响大
    }
    
    # 各道路类型阻塞密度 (单位: PCU/公里)
    k_jam = {1: 150, 2: 120, 3: 100, 4: 70}
    
    # 各道路类型阻塞速度 (km/h)
    jam_speed = {1: 30, 2: 30, 3: 20, 4: 15}
    
    # 权重参数
    alpha = 0.9  # 速度因子权重
    beta = 0.1   # 密度因子权重
    
    # 2. 读取数据
    input_file = r'e:\github_projects\math_modeling\data\02重命名.csv'
    try:
        df = pd.read_csv(input_file)
        print(f"成功读取数据，共 {len(df)} 条记录")
    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_file}")
        return
    except Exception as e:
        print(f"读取数据时发生错误：{e}")
        return
    
    # 3. 计算拥堵指数
    congestion_indices = []
    
    for idx, row in df.iterrows():
        # 第1步：计算总加权拥堵单位 (Total PCU)
        total_pcu = (row['ebike_count'] * pcu_weights['ebike'] + 
                    row['car_count'] * pcu_weights['car'] + 
                    row['bus_count'] * pcu_weights['bus'])
        
        # 第2步：计算加权交通密度 (Weighted Traffic Density)
        if row['road_length_km'] > 0:
            weighted_density = total_pcu / row['road_length_km']
        else:
            weighted_density = 0
        
        # 第3步：计算速度与密度因子
        road_type = int(row['road_type_code'])
        k_jam_value = k_jam.get(road_type, 70)  # 默认值为支路的阻塞密度
        jam_speed_value = jam_speed.get(road_type, 15)  # 默认值为支路的阻塞速度
        
        # 速度因子计算
        speed_factor_raw = max(0, 1 - row['avg_speed_kph'] / jam_speed_value)
        speed_factor = speed_factor_raw ** 0.5
        
        # 密度因子计算 (使用加权密度)
        density_factor_raw = min(1, weighted_density / k_jam_value)
        density_factor = density_factor_raw ** 0.5
        
        # 第4步：加权合成最终指数
        congestion_index = (alpha * speed_factor) + (beta * density_factor)
        
        # 确保指数在[0,1]范围内
        congestion_index = max(0, min(1, congestion_index))
        
        congestion_indices.append(congestion_index)
    
    # 4. 添加拥堵指数列到数据框
    df['congestion_index'] = congestion_indices
    
    # 5. 保存结果
    output_file = r'e:\github_projects\math_modeling\data\03拥挤.csv'
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 保存CSV文件
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"拥堵指数计算完成，结果已保存到：{output_file}")
        
        # 输出统计信息
        print(f"\n拥堵指数统计信息：")
        print(f"最小值: {df['congestion_index'].min():.4f}")
        print(f"最大值: {df['congestion_index'].max():.4f}")
        print(f"平均值: {df['congestion_index'].mean():.4f}")
        print(f"标准差: {df['congestion_index'].std():.4f}")
        
    except Exception as e:
        print(f"保存文件时发生错误：{e}")

if __name__ == "__main__":
    calculate_congestion_index()
