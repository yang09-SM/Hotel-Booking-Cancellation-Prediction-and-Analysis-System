
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    df_processed = df.copy()
    
    # 去除标签泄漏特征（预测时无法获取的特征
    leak_features = ['reservation_status', 'reservation_status_date']
    df_processed = df_processed.drop(columns=leak_features, errors='ignore')
    
    # 处理缺失值
    df_processed['children'] = df_processed['children'].fillna(0)
    df_processed['country'] = df_processed['country'].fillna('Unknown')
    df_processed['agent'] = df_processed['agent'].fillna(0)
    df_processed['company'] = df_processed['company'].fillna(0)
    
    # 类别编码
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le
    
    # 分离特征和标签
    X = df_processed.drop('is_canceled', axis=1)
    y = df_processed['is_canceled']
    
    # 标准化数值特征
    scaler = StandardScaler()
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    return X, y, label_encoders, scaler, categorical_cols, numeric_cols

def train_models(X_train, y_train):
    models = {}
    
    # 1. 逻辑回归
    print("训练逻辑回归模型...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = lr
    
    # 2. 随机森林（集成学习）
    print("训练随机森林模型...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    
    # 3. 支持向量机
    print("训练支持向量机模型...")
    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train, y_train)
    models['SVM'] = svm
    
    # 4. 神经网络
    print("训练神经网络模型...")
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    mlp.fit(X_train, y_train)
    models['Neural Network'] = mlp
    
    return models

def evaluate_models(models, X_test, y_test):
    results = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if hasattr(model, 'predict_proba') else 0.5
        }
    
    return results

def save_models(models, label_encoders, scaler, feature_names, output_dir='models'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存模型
    for name, model in models.items():
        safe_name = name.replace(' ', '_')
        joblib.dump(model, f'{output_dir}/{safe_name}_model.pkl')
    
    # 保存预处理组件
    joblib.dump(label_encoders, f'{output_dir}/label_encoders.pkl')
    joblib.dump(scaler, f'{output_dir}/scaler.pkl')
    joblib.dump(feature_names, f'{output_dir}/feature_names.pkl')
    
    print(f"模型已保存到 {output_dir} 目录")

def main():
    print("开始训练酒店预订取消预测模型...")
    
    # 加载数据
    df = load_data('hotel_bookings.csv')
    print(f"数据加载完成，共 {len(df)} 条记录")
    
    # 预处理
    X, y, label_encoders, scaler, categorical_cols, numeric_cols = preprocess_data(df)
    feature_names = X.columns.tolist()
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"训练集: {len(X_train)} 条, 测试集: {len(X_test)} 条")
    
    # 训练模型
    models = train_models(X_train, y_train)
    
    # 评估模型
    results = evaluate_models(models, X_test, y_test)
    
    # 打印结果
    print("\n模型性能评估结果:")
    print("-" * 80)
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  准确率: {metrics['accuracy']:.4f}")
        print(f"  精确率: {metrics['precision']:.4f}")
        print(f"  召回率: {metrics['recall']:.4f}")
        print(f"  F1分数: {metrics['f1']:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # 保存结果
    joblib.dump(results, 'models/model_results.pkl')
    
    # 保存模型
    save_models(models, label_encoders, scaler, feature_names)
    
    print("\n训练完成！")

if __name__ == "__main__":
    main()

