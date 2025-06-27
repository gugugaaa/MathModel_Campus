import pandas as pd
import os

# 文件路径
input_file = r'data\04早晚高峰.csv'
output_file = r'data\04早晚高峰_with_passenger_throughput.csv'

# 载人量计算的固定参数
P_car_avg = 1.3    # 平均每辆小汽车载客人数
P_ebike_avg = 1.05  # 平均每辆电动车载客人数
P_bus_avg = 60     # 平均每辆公交车载客人数

def calculate_passenger_throughput():
    """
    计算载人量并添加到CSV文件中
    """
    # 读取CSV文件
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(input_file, encoding='gbk')
    
    print(f"原始数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    
    # 检查必需的列是否存在
    required_columns = ['car_count', 'ebike_count', 'bus_count']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"警告：缺少必需的列: {missing_columns}")
        return
    
    # 计算载人量
    df['passenger_throughput'] = (
        P_car_avg * df['car_count'] +
        P_ebike_avg * df['ebike_count'] +
        P_bus_avg * df['bus_count']
    )
    
    # 显示计算结果的统计信息
    print(f"\n载人量统计信息:")
    print(f"最小值: {df['passenger_throughput'].min():.2f}")
    print(f"最大值: {df['passenger_throughput'].max():.2f}")
    print(f"平均值: {df['passenger_throughput'].mean():.2f}")
    print(f"中位数: {df['passenger_throughput'].median():.2f}")
    
    # 保存结果
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n处理完成！结果已保存到: {output_file}")
    print(f"新数据形状: {df.shape}")
    
    # 显示前几行数据
    print(f"\n前5行数据预览:")
    display_columns = ['road_id'] + required_columns + ['passenger_throughput']
    available_columns = [col for col in display_columns if col in df.columns]
    print(df[available_columns].head())

if __name__ == "__main__":
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：输入文件不存在: {input_file}")
    else:
        calculate_passenger_throughput()