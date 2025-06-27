import pandas as pd

# 假设原始csv路径
input_csv = r'data\08聚类\03高拥堵.csv'
# 输出新csv路径
output_csv = r'data\10线性规划\03高拥堵.csv'

# 需要保留的字段
columns_to_keep = [
    'road_id',
    'road_type_code',
    'ebike_count',
    'road_length_km',
    'avg_speed_kph'
]

df = pd.read_csv(input_csv)
print("原始数据形状:", df.shape)
# 删除road_type_code=1（高速公路）的数据
df = df[df['road_type_code'] != 1]
df_slim = df[columns_to_keep]
print("筛选后数据形状:", df_slim.shape)
df_slim.to_csv(output_csv, index=False)