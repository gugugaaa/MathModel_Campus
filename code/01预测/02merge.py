import pandas as pd
import numpy as np

def merge_road_data(input_file, output_file):
    """
    合并路段数据，删除星期信息，按路段ID和时间段聚合数据
    """
    # 读取数据
    df = pd.read_csv(input_file)
    
    # 定义需要保留的列（删除星期相关列）
    columns_to_keep = [
        'ROADSECT_ID', 'ROADSECT_NAME', 'SECT10', 'SECT10_NAME', 
        'TYPE', 'ROADLENGTH', 'TIME_SLOT_30MIN', 'WEEKEND',
        'GOCOUNT', 'SPEED_kph', 'FLOW_vph', 'DENSITY_vpkm', 
        'JAM_SPEED', 'TYPE_NAME', 'est_bus_count', 'est_ebike_count', 
        'est_walk_count', 'est_ebike_speed', 'est_bus_speed', 'est_walk_speed'
    ]
    
    # 选择需要的列
    df_filtered = df[columns_to_keep].copy()
    
    # 定义聚合方式
    agg_dict = {
        'ROADSECT_NAME': 'first',  # 路段名称取第一个
        'SECT10': 'first',         # 区域编号取第一个
        'SECT10_NAME': 'first',    # 区域名称取第一个
        'TYPE': 'first',           # 路段类型取第一个
        'ROADLENGTH': 'first',     # 路段长度取第一个
        'WEEKEND': 'first',        # 是否周末取第一个
        'TYPE_NAME': 'first',      # 类型名称取第一个
        'JAM_SPEED': 'first',      # 拥堵速度取第一个
        
        # 数值型数据取平均值
        'GOCOUNT': 'mean',         # 通行量
        'SPEED_kph': 'mean',       # 速度
        'FLOW_vph': 'mean',        # 流量
        'DENSITY_vpkm': 'mean',    # 密度
        'est_bus_count': 'mean',   # 估计公交数量
        'est_ebike_count': 'mean', # 估计电动车数量
        'est_walk_count': 'mean',  # 估计步行数量
        'est_ebike_speed': 'mean', # 估计电动车速度
        'est_bus_speed': 'mean',   # 估计公交速度
        'est_walk_speed': 'mean'   # 估计步行速度
    }
    
    # 按路段ID和时间段分组聚合
    merged_df = df_filtered.groupby(['ROADSECT_ID', 'TIME_SLOT_30MIN']).agg(agg_dict).reset_index()
    
    # 重新排列列的顺序
    column_order = [
        'ROADSECT_ID', 'ROADSECT_NAME', 'SECT10', 'SECT10_NAME', 
        'TYPE', 'ROADLENGTH', 'TIME_SLOT_30MIN', 'WEEKEND',
        'GOCOUNT', 'SPEED_kph', 'FLOW_vph', 'DENSITY_vpkm', 
        'JAM_SPEED', 'TYPE_NAME', 'est_bus_count', 'est_ebike_count', 
        'est_walk_count', 'est_ebike_speed', 'est_bus_speed', 'est_walk_speed'
    ]
    
    merged_df = merged_df[column_order]
    
    # 保存结果
    merged_df.to_csv(output_file, index=False)
    
    print(f"原始数据行数: {len(df)}")
    print(f"合并后数据行数: {len(merged_df)}")
    print(f"合并后的数据已保存到: {output_file}")
    
    return merged_df

# 使用示例
if __name__ == "__main__":
    # 输入和输出文件路径
    input_file = r"data\01预测.csv"  # 请替换为实际的输入文件路径
    output_file = r"data\01合并预测.csv"
    
    # 执行合并
    result_df = merge_road_data(input_file, output_file)
    
    # 显示前几行结果
    print("\n合并后数据预览:")
    print(result_df.head())