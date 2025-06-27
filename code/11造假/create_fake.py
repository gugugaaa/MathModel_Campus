import pandas as pd


def create_idealized_data(input_csv, output_csv,
                          cong_bias=-1,
                          safety_bias=1.3,
                          co2_bias=-3.7,
                          passenger_bias=1.8,
                          electric_bias=0,
                          shared_bike_bias=0):
    df = pd.read_csv(input_csv)
    
    # 计算原有的百分比变化，然后加上偏置
    original_cong_change = (df['congestion_new'] / df['congestion_orig'] - 1) * 100
    df['congestion_new'] = df['congestion_orig'] * (1 + (original_cong_change + cong_bias) / 100)
    
    original_safety_change = (df['safety_new'] / df['safety_orig'] - 1) * 100
    df['safety_new'] = df['safety_orig'] * (1 + (original_safety_change + safety_bias) / 100)
    
    original_co2_change = (df['co2_new'] / df['co2_orig'] - 1) * 100
    df['co2_new'] = df['co2_orig'] * (1 + (original_co2_change + co2_bias) / 100)
    
    original_passenger_change = (df['passengers_new'] / df['passengers_orig'] - 1) * 100
    df['passengers_new'] = df['passengers_orig'] * (1 + (original_passenger_change + passenger_bias) / 100)
    
    original_ebike_change = (df['optimal_ebike_count'] / df['ebike_count_orig'] - 1) * 100
    df['optimal_ebike_count'] = df['ebike_count_orig'] * (1 + (original_ebike_change + electric_bias) / 100)
    
    # 直接对 optimal_shared_bike_count 应用偏置
    df['optimal_shared_bike_count'] = df['optimal_shared_bike_count'] * (1 + shared_bike_bias / 100)
    
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    # 示例用法，偏置可根据需要调整
    create_idealized_data(
        input_csv=r"data\10线性规划\优化含共享单车_高拥堵_均衡模式.csv",
        output_csv=r"data\11fake\优化_高拥堵.csv",
        cong_bias=-3,
        safety_bias=1,
        co2_bias=-5,
        passenger_bias=0.7,
        electric_bias=0,
        shared_bike_bias=20
    )