import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import warnings
warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')

def load_data(filepath):
    return pd.read_csv(filepath)

def explore_data(df):
    print("=== 数据集基本信息 ===")
    print(df.info())
    print("\n=== 数据集维度 ===")
    print(f"行数: {df.shape[0]}, 列数: {df.shape[1]}")
    print("\n=== 前5行数据 ===")
    print(df.head())
    print("\n=== 统计摘要 ===")
    print(df.describe())
    print("\n=== 类别变量分布 ===")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"\n{col} 分布:")
        print(df[col].value_counts(normalize=True)[:5])
    print("\n=== 缺失值情况 ===")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    return categorical_cols

def preprocess_data(df):
    df = df.copy()
    
    df['total_stays'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    df['total_guests'] = df['adults'] + df['children'] + df['babies']
    
    df['children'] = df['children'].fillna(0)
    df['country'] = df['country'].fillna('Unknown')
    df['agent'] = df['agent'].fillna(0)
    df['company'] = df['company'].fillna(0)
    
    df = df.drop(['reservation_status', 'reservation_status_date', 'assigned_room_type'], axis=1)
    
    label_encoder = LabelEncoder()
    df['hotel'] = label_encoder.fit_transform(df['hotel'])
    
    month_mapping = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                     'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
    df['arrival_date_month'] = df['arrival_date_month'].map(month_mapping)
    
    categorical_cols = ['meal', 'country', 'market_segment', 'distribution_channel', 
                        'reserved_room_type', 'deposit_type', 'customer_type']
    
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    X = df.drop('is_canceled', axis=1)
    y = df['is_canceled']
    
    X = X.fillna(0)
    
    scaler = StandardScaler()
    numerical_cols = ['lead_time', 'arrival_date_week_number', 'arrival_date_day_of_month',
                      'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 
                      'children', 'babies', 'booking_changes', 'days_in_waiting_list', 
                      'adr', 'required_car_parking_spaces', 'total_of_special_requests',
                      'total_stays', 'total_guests']
    
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    results = {
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None,
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    print(f"\n=== {model_name} ===")
    print(f"准确率: {results['accuracy']:.4f}")
    print(f"精确率: {results['precision']:.4f}")
    print(f"召回率: {results['recall']:.4f}")
    print(f"F1分数: {results['f1']:.4f}")
    if results['roc_auc'] is not None:
        print(f"ROC-AUC: {results['roc_auc']:.4f}")
    print("混淆矩阵:")
    print(results['confusion_matrix'])
    print(classification_report(y_test, y_pred))
    
    return results

def compare_models(results):
    results_df = pd.DataFrame(results).sort_values('f1', ascending=False)
    print("\n=== 模型性能对比 ===")
    print(results_df[['model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']].to_string(index=False))
    
    plt.figure(figsize=(12, 6))
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    for metric in metrics:
        plt.plot(results_df['model'], results_df[metric], marker='o', label=metric)
    plt.title('模型性能对比')
    plt.xlabel('模型')
    plt.ylabel('分数')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()
    
    return results_df

def main():
    print("=== 酒店预订取消预测分析 ===")
    print("="*50)
    
    df = load_data('hotel_bookings.csv')
    
    explore_data(df)
    
    X, y = preprocess_data(df)
    
    print(f"\n预处理后特征维度: {X.shape}")
    
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print(f"\n训练集: {X_train.shape}, 测试集: {X_test.shape}")
    print(f"训练集标签分布: {y_train.value_counts(normalize=True).to_dict()}")
    print(f"测试集标签分布: {y_test.value_counts(normalize=True).to_dict()}")
    
    models = [
        ('逻辑回归', LogisticRegression(max_iter=1000)),
        ('决策树', DecisionTreeClassifier(random_state=42)),
        ('随机森林', RandomForestClassifier(random_state=42)),
        ('梯度提升', GradientBoostingClassifier(random_state=42)),
        ('AdaBoost', AdaBoostClassifier(random_state=42)),
        ('K近邻', KNeighborsClassifier()),
        ('朴素贝叶斯', GaussianNB()),
    ]
    
    results = []
    for name, model in models:
        result = evaluate_model(model, X_train, X_test, y_train, y_test, name)
        results.append(result)
    
    results_df = compare_models(results)
    
    best_model_name = results_df.iloc[0]['model']
    print(f"\n最佳模型: {best_model_name}")
    
    return results_df

if __name__ == "__main__":
    results = main()