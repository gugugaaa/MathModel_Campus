import pandas as pd
import os

def convert_csv_to_excel(folder_path):
    """
    将指定文件夹内的所有CSV文件转换为同名的Excel文件。

    Args:
        folder_path (str): 包含CSV文件的文件夹路径。
    """
    print(f"正在处理文件夹：{folder_path}\n")

    # 检查路径是否有效
    if not os.path.isdir(folder_path):
        print(f"错误：路径 '{folder_path}' 不是一个有效的文件夹。请检查路径。")
        return

    csv_files_found = 0
    excel_files_converted = 0
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件是否为CSV格式
        if filename.endswith(".csv"):
            csv_files_found += 1
            csv_filepath = os.path.join(folder_path, filename)
            
            # 构建目标Excel文件名（替换扩展名）
            excel_filename = filename.replace(".csv", ".xlsx")
            excel_filepath = os.path.join(folder_path, excel_filename)

            try:
                # 尝试使用UTF-8编码读取CSV文件
                # 如果遇到UnicodeDecodeError，则尝试GBK编码
                try:
                    df = pd.read_csv(csv_filepath, encoding='utf-8')
                except UnicodeDecodeError:
                    print(f"警告：'{filename}' 可能不是UTF-8编码，尝试GBK编码...")
                    df = pd.read_csv(csv_filepath, encoding='gbk')
                
                # 将DataFrame写入Excel文件
                # index=False 避免将DataFrame的索引也写入Excel
                df.to_excel(excel_filepath, index=False)
                excel_files_converted += 1
                print(f"成功转换：'{filename}' -> '{excel_filename}'")
                
            except Exception as e:
                # 捕获并打印转换过程中可能发生的任何错误
                print(f"转换 '{filename}' 时发生错误：{e}")
    
    # 打印转换结果总结
    if csv_files_found == 0:
        print("未在该文件夹中找到任何CSV文件。")
    else:
        print(f"\n--- 转换完成 ---")
        print(f"共找到 {csv_files_found} 个CSV文件。")
        print(f"成功转换 {excel_files_converted} 个文件。")
        if csv_files_found > excel_files_converted:
            print(f"有 {csv_files_found - excel_files_converted} 个文件转换失败，请查看上方错误信息。")

# --- 使用示例 ---
if __name__ == "__main__":
    test_folder =r"data\11fake" 
    # 如果 'test_csvs' 文件夹不存在，则创建一个
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
        print(f"已创建测试文件夹：'{test_folder}'。请在此文件夹中放置CSV文件进行测试。")
    
    # 假设你已经将CSV文件放入 'test_csvs' 文件夹
    convert_csv_to_excel(test_folder)
