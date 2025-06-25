import pandas as pd
import numpy as np

def add_traffic_estimates_fixed(df, w_walk=0.15, O_car_avg=1.3, region_type='深圳市区', 
                               road_type_mapping=None, add_noise=True, noise_level=0.1):
    """
    修正版：为交通数据DataFrame添加估计交通量和速度列
    
    Parameters:
    -----------
    df : pandas.DataFrame
        包含 GOCOUNT, TYPE, SPEED_kph 列的数据框
    w_walk : float, default=0.15
        步行人次份额，默认15%
    O_car_avg : float, default=1.3
        平均每车人数，默认1.3人/车
    region_type : str, default='深圳市区'
        地区类型，用于确定摩托车比例
    road_type_mapping : dict, optional
        道路类型编码映射字典，如 {1.0: '高速公路', 2.0: '主干道'}
    add_noise : bool, default=True
        是否添加随机噪声增加变异性
    noise_level : float, default=0.1
        噪声水平（相对标准差）
        
    Returns:
    --------
    pandas.DataFrame
        添加了新列的数据框
    """
    
    # 创建数据副本
    result_df = df.copy()
    
    # 处理道路类型编码映射
    if road_type_mapping is None:
        # 根据数据中的实际编码推断映射关系
        unique_types = sorted(df['TYPE'].dropna().unique())
        print(f"检测到道路类型编码: {unique_types}")
        
        # 默认映射（根据常见的道路等级）
        if len(unique_types) == 4:
            road_type_mapping = {
                unique_types[0]: '高速公路',    # 1.0 对应高速公路
                unique_types[1]: '主干道',     # 2.0 对应主干道
                unique_types[2]: '次干道',     # 3.0 对应次干道  
                unique_types[3]: '支路'       # 4.0 对应支路
            }
        else:
            # 通用映射
            road_type_mapping = {t: f'道路类型_{int(t)}' for t in unique_types}
    
    # 创建映射后的道路类型列
    result_df['TYPE_NAME'] = result_df['TYPE'].map(road_type_mapping)
    print(f"道路类型映射: {road_type_mapping}")
    
    # 修正的道路类型系数函数
    def get_bus_ratio(road_type_code):
        """根据道路类型编码返回公交车比例"""
        # 1=高速公路, 2=主干道, 3=次干道, 4=支路
        if road_type_code == 1:
            return 0.015  # 高速公路 1-2%
        elif road_type_code == 2:
            return 0.04   # 主干道 3-5%
        elif road_type_code == 3:
            return 0.02   # 次干道 
        else:  # 4
            return 0.005  # 支路 ≈0%
    
    def get_ebike_ratio(road_type_code):
        """根据道路类型编码返回电动两轮车比例"""
        if road_type_code == 1:
            return 0.0    # 高速公路 0%
        elif road_type_code == 2:
            return 0.06   # 主干道 5%
        elif road_type_code == 3:
            return 0.12   # 次干道 
        else:  # 4
            return 0.15   # 支路 ≥10%
    
    def get_walk_base_ratio(road_type_code):
        """根据道路类型编码返回步行者基础比例系数"""
        if road_type_code == 1:
            return 0   # 高速公路：极少步行者，仅服务区等
        elif road_type_code == 2:
            return 0.15   # 主干道：有人行道但步行者相对较少
        elif road_type_code == 3:
            return 0.35   # 次干道：较多步行者，连接居住区和商业区
        else:  # 4
            return 0.60   # 支路：步行者密度最高，社区内部道路
  
    
    # 添加噪声函数
    def add_multiplicative_noise(values, noise_level):
        """添加乘性噪声"""
        if not add_noise:
            return values
        noise = np.random.lognormal(0, noise_level, len(values))
        return values * noise
    
    # 计算估计公交车数量
    bus_ratios = result_df['TYPE'].apply(get_bus_ratio)
    result_df['est_bus_count'] = result_df['GOCOUNT'] * bus_ratios
    result_df['est_bus_count'] = add_multiplicative_noise(result_df['est_bus_count'], noise_level)
    result_df['est_bus_count'] = result_df['est_bus_count'].round().astype(int)
    
    # 计算估计电动两轮车数量
    ebike_ratios = result_df['TYPE'].apply(get_ebike_ratio)
    result_df['est_ebike_count'] = result_df['GOCOUNT'] * ebike_ratios
    result_df['est_ebike_count'] = add_multiplicative_noise(result_df['est_ebike_count'], noise_level)
    result_df['est_ebike_count'] = result_df['est_ebike_count'].round().astype(int)
  
    
    # 计算估计步行者数量 - 修正版：考虑道路类型特征
    walk_base_ratios = result_df['TYPE'].apply(get_walk_base_ratio)
    # 基于道路容量和道路类型特征计算步行者数量
    walk_base = walk_base_ratios * result_df['GOCOUNT'] * O_car_avg * (w_walk / (1 - w_walk))
    result_df['est_walk_count'] = add_multiplicative_noise(walk_base, noise_level)
    result_df['est_walk_count'] = result_df['est_walk_count'].round().astype(int)
    
    # 计算速度（添加小量随机变异）
    speed_noise = noise_level * 0.5 if add_noise else 0
    
    # 电动车速度
    ebike_speed_base = np.minimum(25, result_df['SPEED_kph'] * 0.8)
    if add_noise:
        speed_variation = np.random.normal(1.0, speed_noise, len(ebike_speed_base))
        ebike_speed_base *= speed_variation
    result_df['est_ebike_speed'] = np.maximum(5, np.minimum(25, ebike_speed_base)).round(1)
    
    # 公交车速度
    bus_speed_base = np.maximum(10, np.minimum(25, result_df['SPEED_kph'] * 0.7))
    if add_noise:
        speed_variation = np.random.normal(1.0, speed_noise, len(bus_speed_base))
        bus_speed_base *= speed_variation
    result_df['est_bus_speed'] = np.maximum(10, np.minimum(25, bus_speed_base)).round(1)
  
    
    # 步行速度（添加小量个体差异）
    if add_noise:
        walk_speed_variation = np.random.normal(5.0, 0.5, len(result_df))
        result_df['est_walk_speed'] = np.maximum(3.0, np.minimum(7.0, walk_speed_variation)).round(1)
    else:
        result_df['est_walk_speed'] = 5.0
      # 输出统计信息
    print(f"\n数据处理完成:")
    print(f"- 道路类型分布:")
    for code, name in road_type_mapping.items():
        count = (result_df['TYPE'] == code).sum()
        pct = count / len(result_df) * 100
        print(f"  {name}({code}): {count:,} ({pct:.1f}%)")
    print(f"\n- 预测结果概览:")
    print(f"  公交车平均数量: {result_df['est_bus_count'].mean():.1f}")
    print(f"  电动车平均数量: {result_df['est_ebike_count'].mean():.1f}")
    print(f"  步行者平均数量: {result_df['est_walk_count'].mean():.1f}")
    
    return result_df


# 使用示例：处理你的实际数据
def process_your_data(df):
    """处理实际数据的便捷函数"""
    
    # 自动检测道路类型编码
    unique_types = sorted(df['TYPE'].dropna().unique())
    print(f"检测到的道路类型编码: {unique_types}")
    
    # 根据你的数据情况定义映射
    # 修正后的映射：
    # 1 - 高速公路
    # 2 - 主干道  
    # 3 - 次干道
    # 4 - 支路
    
    road_mapping = {
        1: '高速公路',
        2: '主干道', 
        3: '次干道',
        4: '支路'
    }
    
    # 处理数据
    processed_df = add_traffic_estimates_fixed(
        df, 
        road_type_mapping=road_mapping,
        w_walk=0.12,  # 步行人次份额12%
        add_noise=True,           # 添加噪声增加真实性
        noise_level=0.15,         # 15%的变异水平
        region_type='深圳'         # 区域类型
    )
    
    return processed_df



# 主程序：加载数据并进行预测
if __name__ == "__main__":
    # 加载数据
    data_path = r"E:\github_projects\math_modeling_playground\playground\day1\02原始降采样\速度_属性_降采样_1周_30分钟.csv"
    
    try:
        print(f"正在加载数据: {data_path}")
        df = pd.read_csv(data_path, encoding='utf-8')
        print(f"数据加载成功，共 {len(df)} 行数据")
        
        # 检查数据结构
        print(f"\n数据列名: {list(df.columns)}")
        print(f"数据前5行:")
        print(df.head())
        
        # 检查必要的列是否存在
        required_cols = ['GOCOUNT', 'TYPE', 'SPEED_kph']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"警告：缺少必要的列: {missing_cols}")
            print("请检查数据文件中的列名是否正确")
        else:
            # 进行交通预测
            print(f"\n开始进行交通量预测...")
            processed_df = process_your_data(df)
            
            # 保存结果
            output_path = r"data\01预测.csv"
            processed_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"\n预测结果已保存到: {output_path}")
            
            # 显示结果概览
            print(f"\n预测结果列名:")
            print(list(processed_df.columns))
            
    except FileNotFoundError:
        print(f"错误：找不到数据文件 {data_path}")
        print("请检查文件路径是否正确")
    except Exception as e:
        print(f"处理数据时出错: {str(e)}")
        import traceback
        traceback.print_exc()