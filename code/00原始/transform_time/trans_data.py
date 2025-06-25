# filepath: e:\github_projects\math_modeling_playground\road_speed_analize\trans_data.py

import pandas as pd
import os

def convert_time_to_date():
    """
    将CSV文件中的TIME字段转换为DATE字段
    删除时间部分（00:00:00），只保留日期
    """
    # 输入文件路径
    input_file = r"E:\github_projects\math_modeling_playground\playground\路段速度数据_时间转换.csv"
    
    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"文件不存在: {input_file}")
        return
    
    try:
        # 读取CSV文件
        print("正在读取CSV文件...")
        df = pd.read_csv(input_file)
        
        # 显示原始数据的前几行
        print("原始数据:")
        print(df.head())
        
        # 将TIME列转换为日期时间格式，然后提取日期部分
        df['TIME'] = pd.to_datetime(df['TIME']).dt.date
        
        # 将列名从TIME改为DATE
        df.rename(columns={'TIME': 'DATE'}, inplace=True)
        
        # 显示转换后的数据
        print("\n转换后的数据:")
        print(df.head())
        
        # 保存修改后的数据，覆盖原文件
        df.to_csv(input_file, index=False)
        print(f"\n数据已成功保存到: {input_file}")
        
    except Exception as e:
        print(f"处理文件时出错: {e}")

if __name__ == "__main__":
    convert_time_to_date()