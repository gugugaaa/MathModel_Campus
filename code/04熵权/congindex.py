import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def calculate_entropy_weights(data, input_cols, output_cols):
    """
    计算熵权法权重
    
    Parameters:
    data: DataFrame, 数据
    input_cols: list, 输入变量列名
    output_cols: list, 输出变量列名
    
    Returns:
    dict: 每个输出变量对应的输入变量权重
    """
    
    # 提取输入变量数据
    X = data[input_cols].copy()
    
    # 数据标准化（0-1标准化）
    scaler = MinMaxScaler()
    X_normalized = pd.DataFrame(
        scaler.fit_transform(X), 
        columns=input_cols, 
        index=X.index
    )
    
    # 避免出现0值（会导致log计算错误）
    X_normalized = X_normalized + 1e-10
    
    results = {}
    
    for target_col in output_cols:
        print(f"\n=== 计算 {target_col} 的熵权 ===")
        
        # 将目标变量与输入变量合并进行分析
        analysis_data = X_normalized.copy()
        
        # 计算比重矩阵
        n, m = analysis_data.shape  # n个样本，m个指标
        
        # 计算每个指标下各样本占该指标的比重
        P = analysis_data.div(analysis_data.sum(axis=0), axis=1)
        
        # 计算熵值
        entropy = np.zeros(m)
        for j in range(m):
            # 计算第j个指标的熵值
            p_ij = P.iloc[:, j].values
            # 避免log(0)的情况
            p_ij = np.where(p_ij > 0, p_ij, 1e-10)
            entropy[j] = -np.sum(p_ij * np.log(p_ij)) / np.log(n)
        
        # 计算熵权
        # 熵权 = (1 - 熵值) / sum(1 - 熵值)
        entropy_weights = (1 - entropy) / np.sum(1 - entropy)
        
        # 存储结果
        weight_dict = dict(zip(input_cols, entropy_weights))
        results[target_col] = weight_dict
        
        # 打印结果
        print(f"各指标的熵值:")
        for col, ent in zip(input_cols, entropy):
            print(f"  {col}: {ent:.4f}")
        
        print(f"各指标的熵权:")
        for col, weight in weight_dict.items():
            print(f"  {col}: {weight:.4f}")
        
        # 验证权重和为1
        print(f"权重总和: {sum(entropy_weights):.4f}")
    
    return results

def analyze_weights(results, input_cols, output_cols):
    """
    分析权重结果，生成权重矩阵和排序
    """
    print("\n" + "="*60)
    print("熵权分析结果汇总")
    print("="*60)
    
    # 创建权重矩阵
    weight_matrix = pd.DataFrame(index=input_cols, columns=output_cols)
    
    for output_col in output_cols:
        for input_col in input_cols:
            weight_matrix.loc[input_col, output_col] = results[output_col][input_col]
    
    print("\n权重矩阵:")
    print(weight_matrix.round(4))
    
    # 计算每个输入变量的平均权重
    avg_weights = weight_matrix.mean(axis=1).sort_values(ascending=False)
    print(f"\n输入变量的平均权重排序:")
    for var, weight in avg_weights.items():
        print(f"  {var}: {weight:.4f}")
    
    # 分析每个输出变量最重要的输入因子
    print(f"\n各输出变量的最重要输入因子:")
    for output_col in output_cols:
        weights = weight_matrix[output_col].astype(float)
        max_weight_var = weights.idxmax()
        max_weight_val = weights.max()
        print(f"  {output_col}: {max_weight_var} (权重: {max_weight_val:.4f})")
    
    return weight_matrix

# 主程序
def main():
    # 示例数据（你需要替换为实际的数据文件路径）
    # 如果你有CSV文件，使用: data = pd.read_csv('your_file.csv')
    
    # 创建示例数据
    sample_data = {
        'road_id': [11301, 11302],
        'district_code': [1, 1],
        'road_type_code': [2, 2],
        'road_length_km': [1.004, 0.715],
        'car_count': [1585.8, 1099.1],
        'ebike_count': [82.8, 55.8],
        'bus_count': [61.7, 45.1],
        'pedestrian_count': [366.9, 249.0],
        'avg_speed_kph': [46.0, 45.73],
        'traffic_flow_vph': [1009.9, 599.0],
        'traffic_density_vpkm': [23.8, 14.3],
        'congestion_index': [0.26511, 0.20607]
    }
    
    data = pd.DataFrame(sample_data)
    
    # 定义输入和输出变量
    input_variables = ['road_type_code', 'road_length_km', 'car_count', 'ebike_count', 'pedestrian_count']
    output_variables = ['avg_speed_kph', 'traffic_flow_vph', 'traffic_density_vpkm', 'congestion_index']
    
    print("交通数据熵权分析")
    print("="*50)
    print(f"输入变量: {input_variables}")
    print(f"输出变量: {output_variables}")
    print(f"数据样本数: {len(data)}")
    
    # 计算熵权
    entropy_results = calculate_entropy_weights(data, input_variables, output_variables)
    
    # 分析结果
    weight_matrix = analyze_weights(entropy_results, input_variables, output_variables)
    
    return entropy_results, weight_matrix

# 如果你有实际的CSV文件，使用下面的代码：
def load_and_analyze_csv(file_path):
    """
    从CSV文件加载数据并进行熵权分析
    """
    # 加载数据
    data = pd.read_csv(file_path)
    
    # 定义变量
    input_variables = ['road_type_code', 'road_length_km', 'car_count', 'ebike_count', 'pedestrian_count']
    output_variables = ['avg_speed_kph', 'traffic_flow_vph', 'traffic_density_vpkm', 'congestion_index']
    
    print("交通数据熵权分析")
    print("="*50)
    print(f"数据文件: {file_path}")
    print(f"数据样本数: {len(data)}")
    print(f"输入变量: {input_variables}")
    print(f"输出变量: {output_variables}")
    
    # 检查数据完整性
    print(f"\n数据预览:")
    print(data[input_variables + output_variables].head())
    
    print(f"\n缺失值检查:")
    missing_values = data[input_variables + output_variables].isnull().sum()
    print(missing_values[missing_values > 0])
    
    # 如果有缺失值，可以选择删除或填充
    if missing_values.sum() > 0:
        print("发现缺失值，将删除包含缺失值的行")
        data = data.dropna(subset=input_variables + output_variables)
        print(f"清理后数据样本数: {len(data)}")
    
    # 计算熵权
    entropy_results = calculate_entropy_weights(data, input_variables, output_variables)
    
    # 分析结果
    weight_matrix = analyze_weights(entropy_results, input_variables, output_variables)
    
    return entropy_results, weight_matrix, data

if __name__ == "__main__":
    # 运行示例
    # results, matrix = main()
    
    # 如果要分析实际CSV文件，取消注释下面的代码并提供文件路径
    results, matrix, data = load_and_analyze_csv(r'data\04早晚高峰.csv')