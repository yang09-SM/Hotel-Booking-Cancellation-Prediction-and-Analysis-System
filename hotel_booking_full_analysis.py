import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')

def load_data(filepath):
    return pd.read_csv(filepath)

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
    
    return X, y, X.columns.tolist()

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
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'model_object': model
    }
    
    return results

def plot_feature_importance(model, feature_names, model_name, top_n=15):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'{model_name} 特征重要性')
        plt.barh(range(top_n), importances[indices], align='center')
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.gca().invert_yaxis()
        plt.xlabel('重要性得分')
        plt.tight_layout()
        plt.savefig(f'{model_name}_feature_importance.png')
        plt.close()
        
        return [(feature_names[i], importances[i]) for i in indices]
    return None

def main():
    print("=== 酒店预订取消预测分析 ===")
    print("="*50)
    
    df = load_data('hotel_bookings.csv')
    
    print("\n=== 数据集基本信息 ===")
    print(f"总记录数: {df.shape[0]}")
    print(f"总特征数: {df.shape[1]}")
    print(f"取消率: {df['is_canceled'].mean():.2%}")
    
    X, y, feature_names = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print(f"\n预处理后特征维度: {X.shape}")
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
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
        print(f"\n训练 {name}...")
        result = evaluate_model(model, X_train, X_test, y_train, y_test, name)
        results.append(result)
        print(f"  F1分数: {result['f1']:.4f}, ROC-AUC: {result['roc_auc']:.4f}")
    
    results_df = pd.DataFrame(results).sort_values('f1', ascending=False)
    
    print("\n" + "="*50)
    print("=== 模型性能对比 ===")
    print(results_df[['model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']].to_string(index=False))
    
    best_model = results_df.iloc[0]
    print(f"\n最佳模型: {best_model['model']}")
    print(f"最佳F1分数: {best_model['f1']:.4f}")
    
    feature_importance = plot_feature_importance(best_model['model_object'], feature_names, best_model['model'])
    
    if feature_importance:
        print("\n=== 特征重要性排名 (前15位) ===")
        for i, (feature, importance) in enumerate(feature_importance[:10], 1):
            print(f"{i}. {feature}: {importance:.4f}")
    
    results_df.to_csv('model_results.csv', index=False)
    print("\n结果已保存到 model_results.csv")
    
    return results_df

if __name__ == "__main__":
    results = main()