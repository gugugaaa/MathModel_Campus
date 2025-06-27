import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 设置matplotlib中文字体以正确显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def perform_pcr(df, predictor_columns, model_name="Default Model"):
    """
    对给定的特征集执行主成分回归（PCR）。
    
    参数:
        df (pd.DataFrame): 包含所有数据的DataFrame。
        predictor_columns (list): 用于本次回归的自变量列名列表。
        model_name (str): 用于在输出中标识模型的名称。
        
    返回:
        tuple: 包含R方、实际值和预测值。
    """
    print(f"\n--- 正在运行模型: {model_name} ---")
    print(f"使用的特征数量: {len(predictor_columns)}")
    # print(f"特征列表: {predictor_columns}")

    # 1. 定义自变量（X）和因变量（y）
    target_column = 'congestion_index'
    
    # 清理数据，确保没有无穷大或NaN值
    df_predictors = df[predictor_columns].replace([np.inf, -np.inf], np.nan).dropna()
    X = df_predictors
    y = df.loc[X.index, target_column] # 保证y与X对齐
    
    # 如果数据不足，则无法继续
    if X.shape[0] < 2:
        print("错误：有效数据不足，无法进行建模。")
        return None, None, None

    # 2. 标准化自变量
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    # 3. 主成分分析（PCA）
    pca = PCA(n_components=0.95) 
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"选择的主成分数量: {pca.n_components_}")
    
    # 4. 用主成分拟合线性回归模型
    pcr_model = LinearRegression()
    pcr_model.fit(X_pca, y)
    
    # 5. 预测并计算R方
    y_pred = pcr_model.predict(X_pca)
    r2 = r2_score(y, y_pred)
    
    print(f"模型 R方 (R-squared): {r2:.4f}")

    return r2, y, y_pred

def plot_pcr_results(y_true, y_pred, r2, model_name):
    """
    可视化单次PCR运行的结果。
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.7, edgecolors='k', label='预测值 vs 实际值')
    
    lims = [
        np.min([y_true.min(), y_pred.min()]),
        np.max([y_true.max(), y_pred.max()]),
    ]
    plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='理想预测 (y=x)')

    plt.xlabel("实际拥堵指数")
    plt.ylabel("预测拥堵指数")
    plt.title(f"模型: {model_name}\n实际值与预测值对比 (R²: {r2:.4f})")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

# --- 主程序入口 ---
if __name__ == "__main__":
    # --- 1. 数据加载与特征工程 ---
    try:
        # 尝试加载真实数据，请确保CSV文件路径正确
        csv_path = r"E:\github_projects\math_modeling\data\03拥挤.csv"
        df = pd.read_csv(csv_path, encoding='utf-8')
        # 确保真实数据包含 road_type_code，否则添加一列模拟数据
        if 'road_type_code' not in df.columns:
            print("警告: 真实数据中缺少 'road_type_code'，将为其生成随机值。")
            df['road_type_code'] = np.random.randint(1, 5, size=len(df))

    except FileNotFoundError:
        print(f"错误：未找到文件 {csv_path}")
        print("将使用随机生成的数据进行演示。")
        # --- 生成随机数据用于演示 ---
        n_samples = 200
        data = {
            'road_length_km': np.random.uniform(0.5, 10, n_samples),
            'car_count': np.random.randint(50, 500, n_samples),
            'ebike_count': np.random.randint(20, 300, n_samples),
            'bus_count': np.random.randint(5, 50, n_samples),
            'pedestrian_count': np.random.randint(10, 200, n_samples),
            'avg_speed_kph': np.random.uniform(10, 60, n_samples),
            'road_type_code': np.random.randint(1, 5, n_samples), # 1:高速, 2:主干, 3:次干, 4:支路
        }
        df = pd.DataFrame(data)
        
        # 模拟一个更真实的拥堵指数
        road_type_factor = df['road_type_code'].replace({1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0})
        df['congestion_index'] = (
            (df['car_count'] * 2 + df['ebike_count'] * 0.5 + df['bus_count'] * 3) 
            / df['road_length_km'] * road_type_factor / 100 
            - (df['avg_speed_kph'] / 10) 
            + np.random.randn(n_samples) * 1.5
        )
        # 将拥堵指数缩放到 0-10 范围
        df['congestion_index'] = np.clip(df['congestion_index'], 0, None)
        df['congestion_index'] = (df['congestion_index'] - df['congestion_index'].min())
        df['congestion_index'] = (df['congestion_index'] / df['congestion_index'].max()) * 10
        print("\n注意：当前使用随机生成的数据进行演示。")

    # --- 2. 派生特征创建 ---
    # a) 创建密度特征
    vehicle_types = ['car', 'ebike', 'bus', 'pedestrian']
    epsilon = 1e-6
    df['road_length_km_safe'] = df['road_length_km'].replace(0, epsilon)
    for v_type in vehicle_types:
        df[f'{v_type}_density'] = df[f'{v_type}_count'] / df['road_length_km_safe']

    # b) 对道路类型进行独热编码
    df_road_types = pd.get_dummies(df['road_type_code'], prefix='road_type')
    df = pd.concat([df, df_road_types], axis=1)

    # --- 3. 定义两组特征并执行模型 ---
    
    # 模型一：仅使用密度和速度
    predictors_v1 = [f'{v_type}_density' for v_type in vehicle_types] + ['avg_speed_kph']
    r2_v1, y_actual_v1, y_pred_v1 = perform_pcr(df, predictors_v1, model_name="模型一：基于密度和速度")

    # 模型二：密度、速度 + 道路类型
    # 获取独热编码后的列名
    road_type_cols = [col for col in df.columns if 'road_type_' in str(col)]
    predictors_v2 = predictors_v1 + road_type_cols
    r2_v2, y_actual_v2, y_pred_v2 = perform_pcr(df, predictors_v2, model_name="模型二：基于密度、速度和道路类型")

    # --- 4. 结果对比与可视化 ---
    print("\n" + "="*40)
    print("--- 模型性能对比 ---")
    print(f"模型一 (仅密度) 的 R-squared: {r2_v1:.4f}" if r2_v1 is not None else "模型一未能运行")
    print(f"模型二 (密度+道路类型) 的 R-squared: {r2_v2:.4f}" if r2_v2 is not None else "模型二未能运行")
    print("="*40 + "\n")

    if r2_v2 is not None and r2_v1 is not None:
        if r2_v2 > r2_v1:
            print("结论: 增加'道路类型'特征显著提升了模型的预测能力。")
        else:
            print("结论: 在此数据上，'道路类型'特征未带来明显提升。")

    # 可视化两个模型的结果
    if y_actual_v1 is not None:
        plot_pcr_results(y_actual_v1, y_pred_v1, r2_v1, "模型一：基于密度和速度")
    if y_actual_v2 is not None:
        plot_pcr_results(y_actual_v2, y_pred_v2, r2_v2, "模型二：基于密度、速度和道路类型")
