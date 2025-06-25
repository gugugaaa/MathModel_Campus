import pandas as pd
import os

def period_to_time(period):
    """
    将PERIOD转换为TIME格式
    PERIOD: 1-288，每5分钟一个时间段，从0:00开始
    返回: HH:MM格式的时间字符串
    """
    # PERIOD从1开始，所以要减1
    minutes_from_start = (period - 1) * 5
    
    # 计算小时和分钟
    hours = minutes_from_start // 60
    minutes = minutes_from_start % 60
    
    # 处理跨日情况（24小时制）
    hours = hours % 24
    
    return f"{hours:02d}:{minutes:02d}"

def convert_csv_period_to_time(file_path):
    """
    读取CSV文件，将PERIOD列转换为TIME列，直接在原表上修改
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        print(f"原始数据前5行:")
        print(df.head())
        print(f"\n数据形状: {df.shape}")
        
        # 检查是否存在PERIOD列
        if 'PERIOD' not in df.columns:
            print("错误: 未找到PERIOD列")
            return
        
        # 将PERIOD转换为TIME
        df['TIME'] = df['PERIOD'].apply(period_to_time)
        
        print(f"\n转换后的数据前5行:")
        print(df.head())
        
        # 保存回原文件
        df.to_csv(file_path, index=False)
        print(f"\n已成功将转换结果保存到原文件: {file_path}")
        
        # 显示一些转换示例
        print(f"\n转换示例:")
        for period in [1, 116, 187, 288]:
            if period in df['PERIOD'].values:
                time_str = period_to_time(period)
                print(f"PERIOD {period} -> TIME {time_str}")
        
    except FileNotFoundError:
        print(f"错误: 文件未找到 {file_path}")
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    # CSV文件路径
    csv_file_path = r"E:\github_projects\math_modeling_playground\playground\路段速度数据_时间转换.csv"
    
    # 执行转换
    convert_csv_period_to_time(csv_file_path)