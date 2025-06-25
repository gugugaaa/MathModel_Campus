import pandas as pd
import numpy as np

def process_csv():
    # 读取原始CSV文件
    input_file = r"data\01合并预测.csv"
    output_file = r"data\02重命名.csv"
    
    # 读取数据
    df = pd.read_csv(input_file)
    
    # 创建新的DataFrame，只保留需要的列并重命名
    new_df = pd.DataFrame()
    
    # 映射字段
    field_mapping = {
        'ROADSECT_ID': 'road_id',
        'TYPE': 'road_type_code', 
        'ROADLENGTH': 'road_length_km',
        'GOCOUNT': 'car_count',
        'est_ebike_count': 'ebike_count',
        'est_bus_count': 'bus_count',
        'est_walk_count': 'pedestrian_count',
        'SPEED_kph': 'avg_speed_kph',
        'FLOW_vph': 'traffic_flow_vph',
        'DENSITY_vpkm': 'traffic_density_vpkm',
        'SECT10': 'district_code',
        'TIME_SLOT_30MIN': 'time_slot_30min'
    }
    
    # 按照逻辑顺序复制并重命名字段
    # 定义字段的显示顺序：基本信息 -> 车辆数量 -> 交通指标 -> 时间信息
    ordered_fields = [
        'road_id',           # 路段ID
        'district_code',     # 区域编码
        'road_type_code',    # 道路类型编码
        'road_length_km',    # 道路长度
        'time_slot_30min',   # 30分钟时间段
        'car_count',         # 汽车数量
        'bus_count',         # 公交车数量
        'ebike_count',       # 电动车数量
        'pedestrian_count',  # 步行者数量
        'avg_speed_kph',     # 平均速度
        'traffic_flow_vph',  # 交通流量
        'traffic_density_vpkm' # 交通密度
        
    ]
    
    # 按顺序复制字段
    for new_name in ordered_fields:
        # 找到对应的原字段名
        old_name = None
        for old, new in field_mapping.items():
            if new == new_name:
                old_name = old
                break
        
        if old_name and old_name in df.columns:
            new_df[new_name] = df[old_name].copy()
    
    # 数据处理
    # 删除road_type_code为NaN的路段
    initial_count = len(new_df)
    new_df = new_df.dropna(subset=['road_type_code'])
    removed_count = initial_count - len(new_df)
    # 将road_length_km从米转换为千米
    new_df['road_length_km'] = (new_df['road_length_km'] / 1000).round(3)
    # 确保road_type_code为整数
    new_df['road_type_code'] = new_df['road_type_code'].astype(int)
    # 确保district_code为整数
    new_df['district_code'] = new_df['district_code'].astype(int)
    # 保留指定的小数位数
    new_df['avg_speed_kph'] = new_df['avg_speed_kph'].round(3)
    new_df['traffic_flow_vph'] = new_df['traffic_flow_vph'].round(1)
    new_df['traffic_density_vpkm'] = new_df['traffic_density_vpkm'].round(3)
    
    # 保存处理后的数据
    new_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"数据处理完成！")
    print(f"原始数据形状: {df.shape}")
    print(f"处理后数据形状: {new_df.shape}")
    print(f"保留的字段: {list(new_df.columns)}")
    print(f"输出文件: {output_file}")
    
    # 显示前几行数据预览
    print("\n处理后数据预览:")
    print(new_df.head())

if __name__ == "__main__":
    process_csv()