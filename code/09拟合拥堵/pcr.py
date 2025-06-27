import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def perform_pcr(df):
    """
    执行主成分回归（PCR）以预测拥堵指数。
    
    参数:
        df (pd.DataFrame): 输入数据表。
        
    返回:
        tuple: 包含训练好的线性回归模型、R方值、回归表达式、实际与预测值等。
    """
    # 1. 定义自变量（X）和因变量（y）
    predictor_columns = [
        'road_length_km', 'car_count', 'ebike_count', 
        'bus_count', 'pedestrian_count','avg_speed_kph'
    ]
    target_column = 'congestion_index'
    
    X = df[predictor_columns].dropna()
    y = df.loc[X.index, target_column] # 保证y与X对齐
    
    # 2. 标准化自变量
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    # 3. 主成分分析（PCA）
    # 选择主成分数量，通常选择累计解释方差较高（如95%）的主成分
    pca = PCA(n_components=0.95) 
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"选择的主成分数量: {pca.n_components_}")
    
    # 4. 用主成分拟合线性回归模型
    pcr_model = LinearRegression()
    pcr_model.fit(X_pca, y)
    
    # 5. 预测
    y_pred = pcr_model.predict(X_pca)
    
    # 计算R方
    r2 = r2_score(y, y_pred)
    
    # 6. 构建回归方程
    # 获取主成分的回归系数
    beta = pcr_model.coef_
    # 截距
    beta0 = pcr_model.intercept_
    
    # 获取主成分载荷
    pc_loadings = pca.components_
    
    # 回归方程（主成分表示）
    expression = f"拥堵指数 = {beta0:.4f}"
    for i, coef in enumerate(beta):
        expression += f" + ({coef:.4f} × PC{i+1})"
        
    print("\n--- 主成分回归结果 ---")
    print(f"R方: {r2:.4f}")
    print("回归方程（主成分表示）:")
    print(expression)
    
    # 输出每个主成分的表达式
    print("\n--- 主成分表达式（标准化变量）---")
    for i in range(pca.n_components_):
        terms = []
        for j, col in enumerate(predictor_columns):
            coef = pc_loadings[i, j]
            terms.append(f"{coef:+.4f}×{col}")
        pc_expr = " ".join(terms)
        print(f"PC{i+1} = {pc_expr}")

    # 计算最终回归系数（拥堵指数对原始变量的表达式，变量为标准化后的）
    # 系数 = 各PC系数 * 各PC的载荷，累加
    final_coefs = np.dot(beta, pc_loadings)
    print("\n--- 拥堵指数对原始变量（标准化后）的表达式 ---")
    terms = []
    for coef, col in zip(final_coefs, predictor_columns):
        terms.append(f"{coef:+.4f}×{col}")
    final_expr = f"拥堵指数 = {beta0:.4f} " + " ".join(terms)
    print(final_expr)

    return pcr_model, r2, expression, y, y_pred, pca.n_components_

def plot_pcr_results(y_true, y_pred, r2, expression, n_components):
    """
    使用matplotlib直接可视化PCR结果。
    
    参数:
        y_true (pd.Series): 实际目标值。
        y_pred (np.ndarray): PCR模型预测的目标值。
        r2 (float): 模型的R方值。
        expression (str): 回归方程字符串。
        n_components (int): 使用的主成分数量。
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, edgecolors='k', label='预测值 vs 实际值')
    
    # 添加理想预测（y=x）参考线
    lims = [
        np.min([plt.xlim(), plt.ylim()]),  # 两轴最小值
        np.max([plt.xlim(), plt.ylim()]),  # 两轴最大值
    ]
    plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='理想预测')

    plt.xlabel("实际拥堵指数")
    plt.ylabel("预测拥堵指数")
    plt.title("PCR：实际值与预测值对比")
    plt.legend()
    plt.grid(True)
    
    # 添加回归方程和R方的信息
    info_text = (
        f"使用的主成分数量: {n_components}\n"
        f"R方: {r2:.4f}\n"
        f"回归方程:\n{expression}"
    )
    plt.gcf().text(0.02, 0.02, info_text, fontsize=10, va='bottom', ha='left')
    plt.tight_layout()
    plt.show()

# --- 主程序入口 ---
if __name__ == "__main__":
    # 加载数据
    # 注意：请确保CSV文件路径正确
    try:
        csv_path = r"E:\github_projects\math_modeling\data\03拥挤.csv"
        df = pd.read_csv(csv_path, encoding='utf-8')
    except FileNotFoundError:
        print(f"错误：未找到文件 {csv_path}")
        print("请修改 'csv_path' 变量为正确的文件位置。")
        # 若找不到文件，则生成随机数据用于演示
        data = {
            'road_length_km': np.random.rand(100) * 10,
            'car_count': np.random.randint(50, 500, 100),
            'ebike_count': np.random.randint(20, 300, 100),
            'bus_count': np.random.randint(5, 50, 100),
            'pedestrian_count': np.random.randint(10, 200, 100),
            'congestion_index': np.random.rand(100) * 10
        }
        df = pd.DataFrame(data)
        print("\n注意：当前使用随机生成的数据进行演示。")


    # 执行主成分回归
    model, r2, expression, y_actual, y_predicted, n_components = perform_pcr(df)
    
    # 可视化结果
    plot_pcr_results(y_actual, y_predicted, r2, expression, n_components)