import pandas as pd
import numpy as np

def merge_rush_hour_data(csv_file_path):
    """
    合并早晚高峰数据，计算每条路段的平均一小时数据
    """
    # 读取数据
    df = pd.read_csv(csv_file_path)
    
    # 定义早晚高峰时间段
    morning_rush = ['07:00', '07:30', '08:00', '08:30', '09:00']  # 早高峰 7:00-9:00
    evening_rush = ['17:00', '17:30', '18:00', '18:30', '19:00']  # 晚高峰 17:00-19:00
    
    # 筛选早晚高峰数据
    rush_hour_data = df[df['time_slot_30min'].isin(morning_rush + evening_rush)]
    
    # 按路段ID分组，计算各指标的平均值
    merged_data = rush_hour_data.groupby('road_id').agg({
        'district_code': 'first',  # 取第一个值（区域编码不变）
        'road_type_code': 'first',  # 取第一个值（道路类型不变）
        'road_length_km': 'first',  # 取第一个值（道路长度不变）
        'car_count': 'mean',  # 平均汽车数量
        'ebike_count': 'mean',  # 平均电动车数量
        'bus_count': 'mean',  # 平均公交车数量
        'pedestrian_count': 'mean',  # 平均步行者数量
        'avg_speed_kph': 'mean',  # 平均速度
        'traffic_flow_vph': 'mean',  # 平均交通流量
        'traffic_density_vpkm': 'mean',  # 平均交通密度
        'congestion_index': 'mean'  # 平均拥挤指数
    }).reset_index()
    
    # 将30分钟数据转换为1小时数据（车辆数量相关指标乘以2）
    merged_data['car_count'] = (merged_data['car_count'] * 2).round(1)
    merged_data['ebike_count'] = (merged_data['ebike_count'] * 2).round(1)
    merged_data['bus_count'] = (merged_data['bus_count'] * 2).round(1)
    merged_data['pedestrian_count'] = (merged_data['pedestrian_count'] * 2).round(1)
    
    # 速度、密度和拥挤指数不需要乘以2，保持原值
    merged_data['avg_speed_kph'] = merged_data['avg_speed_kph'].round(2)
    merged_data['traffic_flow_vph'] = merged_data['traffic_flow_vph'].round(1)  # 流量已经是辆/h，不需要乘2
    merged_data['traffic_density_vpkm'] = merged_data['traffic_density_vpkm'].round(1)
    merged_data['congestion_index'] = merged_data['congestion_index'].round(5)
    
    return merged_data

def save_merged_data(input_file, output_file):
    """
    处理数据并保存结果
    """
    try:
        # 合并数据
        merged_df = merge_rush_hour_data(input_file)
        
        # 保存结果
        merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"数据合并完成！")
        print(f"原始数据路段数量可能重复，合并后唯一路段数量: {len(merged_df)}")
        print(f"结果已保存到: {output_file}")
        
        # 显示前几行数据
        print("\n合并后的数据预览:")
        print(merged_df.head())
        
        return merged_df
        
    except Exception as e:
        print(f"处理数据时出错: {e}")
        return None

if __name__ == "__main__":
    # 输入文件路径
    input_file = r"data\03拥挤.csv"
    
    # 输出文件路径
    output_file = r"data\04早晚高峰.csv"
    
    # 执行合并
    result = save_merged_data(input_file, output_file)