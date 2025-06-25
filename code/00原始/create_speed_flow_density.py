import pandas as pd
import os

def process_traffic_data(input_file, output_file):
    """
    处理交通数据，计算速度、流量和密度
    """
    # 读取CSV文件
    df = pd.read_csv(input_file)
      # 计算新字段
    # 平均速度(km/h) = (GOLEN / GOTIME) * 3.6
    df['SPEED_kph'] = ((df['GOLEN'] / df['GOTIME']) * 3.6).round(3)
    
    # 时间片时长为5分钟 = 5/60 小时
    time_interval_hours = 5 / 60
    
    # 平均流量(辆/h) = GOCOUNT / 时间片时长_小时
    df['FLOW_vph'] = (df['GOCOUNT'] / time_interval_hours).round(3)
    
    # 平均密度(辆/km) = FLOW_vph / SPEED_kph
    df['DENSITY_vpkm'] = (df['FLOW_vph'] / df['SPEED_kph']).round(3)
    
    # 删除原始字段GOLEN和GOTIME
    df = df.drop(['GOLEN', 'GOTIME'], axis=1)
    
    # 重新排列列的顺序，让新字段在合适的位置
    columns_order = ['ROADSECT_ID', 'GOCOUNT', 'DATE', 'TIME', 'WEEKEND', 
                    'SPEED_kph', 'FLOW_vph', 'DENSITY_vpkm']
    df = df[columns_order]
    
    # 保存处理后的数据
    df.to_csv(output_file, index=False)
    
    print(f"数据处理完成！")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"处理了 {len(df)} 行数据")
    
    # 显示前几行数据预览
    print("\n处理后的数据预览:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    # 输入和输出文件路径
    input_file = r"E:\github_projects\math_modeling_playground\playground\路段速度数据_01.csv"
    output_file = r"E:\github_projects\math_modeling_playground\playground\路段速度数据_02.csv"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：输入文件 {input_file} 不存在！")
    else:
        # 处理数据
        processed_df = process_traffic_data(input_file, output_file)