import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

# ====================== 自动扫描文件夹内所有csv，万能读取，彻底解决文件名报错 ======================
# 获取当前test.py所在文件夹
current_dir = os.path.dirname(os.path.abspath(__file__))
# 遍历文件夹，找到所有.csv结尾的文件
csv_files = [f for f in os.listdir(current_dir) if f.endswith('.csv')]

print("当前文件夹里所有csv文件：", csv_files)
# 自动读取第一个csv数据集
csv_name = csv_files[0]
csv_path = os.path.join(current_dir, csv_name)
print(f"自动匹配到数据集：{csv_name}")

# 读取数据
df = pd.read_csv(csv_path)
print("✅ 数据集读取成功！")
print("原始数据形状（行，列）：", df.shape)

# ====================== 2. 查看原始类别型字段 ======================
print("\n===== 原始所有类别型变量（object文本字段） =====")
cat_origin_cols = df.select_dtypes(include=['object']).columns.tolist()
print(cat_origin_cols)

# ====================== 3. 剔除泄露、无用、高基数字段 ======================
drop_cols = [
    'reservation_status',
    'reservation_status_date',
    'country',
    'agent',
    'company'
]
# 筛选出本次需要数值化的有效类别列
cat_cols = [col for col in cat_origin_cols if col not in drop_cols]
print("\n===== 本次需要预处理的类别字段 =====")
print(cat_cols)

# ====================== 4. 类别预处理：编码转数值 ======================
# 思路1：arrival_date_month 入住月份【有序分类】→ 标签编码 LabelEncoder
le = LabelEncoder()
df['arrival_date_month'] = le.fit_transform(df['arrival_date_month'])
print("\n月份标签编码对照表：")
print(dict(zip(le.classes_, le.transform(le.classes_))))

# 思路2：其余所有【无序分类】→ 独热编码 One-Hot Encoding，转0/1数值
one_hot_cols = [col for col in cat_cols if col != 'arrival_date_month']
df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)

# ====================== 5. 预处理结果输出 ======================
print("\n" + "="*60)
print("🎉 类别数据预处理全部完成！")
print("编码后数据集形状（行，列）：", df.shape)
print("编码后所有字段均为数值型，可直接用于机器学习建模")
print("\n编码后前5行数据预览：")
print(df.head())