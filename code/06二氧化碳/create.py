import pandas as pd
import os

# 文件路径
input_file = r'data\05载人量.csv'
output_file = r'data\06二氧化碳.csv'

# 二氧化碳排放计算的固定参数
EF_car_petrol = 0.131  # kg CO₂ / 车·km（汽油车）
ratio_petrol  = 0.754  # 小汽车中汽油车占比
EF_car_elec   = 0.074  # kg CO₂ / 车·km（电动小汽车）
ratio_elec    = 0.246  # 小汽车中电动占比
EF_ebike      = 0.001 # kg CO₂ / 车·km（电动自行车）
EF_bus        = 0.44   # kg CO₂ / 车·km（深圳公交为纯电）

def calculate_co2_emission():
    """
    计算二氧化碳排放量并添加到CSV文件中
    """
    # 读取CSV文件
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(input_file, encoding='gbk')
    
    print(f"原始数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    
    # 检查必需的列是否存在
    required_columns = ['car_count', 'ebike_count', 'bus_count', 'road_length_km']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"警告：缺少必需的列: {missing_columns}")
        return
    
    # 计算二氧化碳排放量
    df['co2_emission'] = (
        EF_car_petrol * ratio_petrol * df['car_count'] +
        EF_car_elec   * ratio_elec   * df['car_count'] +
        EF_ebike * df['ebike_count'] +
        EF_bus   * df['bus_count']
    ) * df['road_length_km']
    
    # 显示计算结果的统计信息
    print(f"\n二氧化碳排放量统计信息:")
    print(f"最小值: {df['co2_emission'].min():.4f} kg")
    print(f"最大值: {df['co2_emission'].max():.4f} kg")
    print(f"平均值: {df['co2_emission'].mean():.4f} kg")
    print(f"中位数: {df['co2_emission'].median():.4f} kg")
    print(f"总排放量: {df['co2_emission'].sum():.2f} kg")
    
    # 按道路类型统计
    if 'road_type_code' in df.columns:
        print(f"\n按道路类型统计二氧化碳排放量:")
        road_type_names = {1: '高速公路', 2: '主干道', 3: '次干道', 4: '支路'}
        # 创建临时的道路类型名称用于统计显示
        road_type_temp = df['road_type_code'].map(road_type_names)
        co2_by_type = df.groupby(road_type_temp)['co2_emission'].agg(['sum', 'mean', 'count'])
        print(co2_by_type.round(4))
    
    # 保存结果
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n处理完成！结果已保存到: {output_file}")
    print(f"新数据形状: {df.shape}")
    
    # 显示前几行数据
    print(f"\n前5行数据预览:")
    display_columns = ['road_id'] + required_columns + ['co2_emission']
    if 'passenger_throughput' in df.columns:
        display_columns.insert(-1, 'passenger_throughput')
    available_columns = [col for col in display_columns if col in df.columns]
    print(df[available_columns].head())

if __name__ == "__main__":
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：输入文件不存在: {input_file}")
    else:
        calculate_co2_emission()