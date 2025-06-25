# filepath: e:\github_projects\math_modeling_playground\trans_time.py
import pandas as pd
from datetime import datetime

def convert_time_column(file_path, output_path=None, encoding='utf-8'):
    """
    转换CSV文件中的TIME列（Unix毫秒时间戳）为北京时间
    
    Parameters:
    -----------
    file_path : str
        输入CSV文件路径
    output_path : str, optional
        输出文件路径，如果不指定则覆盖原文件
    encoding : str
        文件编码，默认utf-8
    
    Returns:
    --------
    pd.DataFrame : 转换后的数据框
    """
    try:
        # 读取数据
        df = pd.read_csv(file_path, encoding=encoding)
        print(f"成功读取数据，共 {len(df):,} 行")
        
        # 检查TIME列是否存在
        if 'TIME' not in df.columns:
            print("错误：未找到TIME列")
            return None
        
        # 备份原始TIME列
        df['TIME_ORIGINAL'] = df['TIME']
        
        # 转换Unix毫秒时间戳为北京时间
        df['TIME'] = (pd.to_datetime(df['TIME'], unit='ms', utc=True)
                      .dt.tz_convert('Asia/Shanghai')
                      .dt.strftime('%Y-%m-%d %H:%M:%S'))
        
        print("时间转换完成")
        print(f"时间范围: {df['TIME'].min()} 至 {df['TIME'].max()}")
        
        # 保存结果
        if output_path is None:
            output_path = file_path
        
        df.to_csv(output_path, index=False, encoding=encoding)
        print(f"结果已保存到: {output_path}")
        
        return df
        
    except Exception as e:
        print(f"处理失败: {e}")
        return None

def preview_time_conversion(file_path, n_rows=10, encoding='utf-8'):
    """
    预览时间转换结果（不保存文件）
    
    Parameters:
    -----------
    file_path : str
        输入CSV文件路径
    n_rows : int
        预览行数
    encoding : str
        文件编码
    """
    try:
        # 读取前n行数据
        df = pd.read_csv(file_path, encoding=encoding, nrows=n_rows)
        
        if 'TIME' not in df.columns:
            print("错误：未找到TIME列")
            return
        
        # 显示原始时间戳
        print("原始时间戳示例:")
        print(df['TIME'].head())
        
        # 转换时间戳
        converted_time = (pd.to_datetime(df['TIME'], unit='ms', utc=True)
                         .dt.tz_convert('Asia/Shanghai')
                         .dt.strftime('%Y-%m-%d %H:%M:%S'))
        
        print("\n转换后的时间:")
        print(converted_time.head())
        
        # 显示对比
        print("\n转换对比:")
        comparison = pd.DataFrame({
            '原始时间戳': df['TIME'],
            '转换后时间': converted_time
        })
        print(comparison)
        
    except Exception as e:
        print(f"预览失败: {e}")

# 使用示例
if __name__ == "__main__":
    # 文件路径
    input_file = 'data/路段速度数据_完整.csv'
    output_file = 'playground/路段速度数据_时间转换.csv'
    
    # 预览转换效果
    print("=== 预览转换效果 ===")
    preview_time_conversion(input_file)
    
    print("\n" + "="*50 + "\n")
    
    # 执行完整转换
    print("=== 执行完整转换 ===")
    result = convert_time_column(input_file, output_file)
    
    if result is not None:
        print("\n转换后数据预览:")
        print(result[['TIME_ORIGINAL', 'TIME']].head())