import pandas as pd
import os

# ===================== 自动读取同目录csv，解决路径问题 =====================
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "hotel_bookings.csv")
df = pd.read_csv(csv_path)

# ===================== 选出所有类别型变量（object类型） =====================
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
print("类别型变量：")
print(cat_cols)
print("-" * 60)

# ===================== 复制两份数据，分别做两种编码 =====================
df_label = df.copy()  # 用于 Label Encoding
df_onehot = df.copy() # 用于 One-Hot Encoding

# ===================== 1. Label Encoding（标签编码）=====================
print("【1】开始 Label Encoding...")
for col in cat_cols:
    df_label[col] = df_label[col].astype("category").cat.codes

print("Label 编码完成！前3行类别列预览：")
print(df_label[cat_cols].head(3))
print("-" * 60)

# ===================== 2. One-Hot Encoding（独热编码）=====================
print("【2】开始 One-Hot Encoding...")
df_onehot = pd.get_dummies(df_onehot, columns=cat_cols, drop_first=True)

print("One-Hot 编码完成！生成的新特征数量：", len(df_onehot.columns))
print("One-Hot 编码后前3行预览（部分列）：")
print(df_onehot.head(3))
print("-" * 60)

# ===================== 结果对比 =====================
print("【编码结果对比】")
print(f"原始特征数：{len(df.columns)}")
print(f"Label 编码后特征数：{len(df_label.columns)}")
print(f"One-Hot 编码后特征数：{len(df_onehot.columns)}")
print("\n✅ 对比完成！")