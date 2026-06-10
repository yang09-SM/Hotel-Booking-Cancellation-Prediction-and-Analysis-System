
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from feature_engineering import FeatureEngineer
from temporal_features import TemporalFeatureEngineer
from feature_interaction import FeatureInteractionDiscovery
from external_data import ExternalDataIntegrator
import warnings
warnings.filterwarnings('ignore')

# 检查 GPU 可用性
def check_gpu_availability():
    try:
        import xgboost as xgb
        # 检查 XGBoost GPU
        xgb_clf = xgb.XGBClassifier(tree_method='hist', device='cuda', n_estimators=1)
        xgb_clf.fit(np.array([[1,2], [3,4]]), np.array([0,1]))
        xgb_available = True
    except:
        xgb_available = False
    
    try:
        import lightgbm as lgb
        # 检查 LightGBM GPU
        lgb_clf = lgb.LGBMClassifier(device='gpu', n_estimators=1, verbose=-1)
        lgb_clf.fit(np.array([[1,2], [3,4]]), np.array([0,1]))
        lgb_available = True
    except:
        lgb_available = False
    
    return xgb_available, lgb_available

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    df_processed = df.copy()
    
    # ===== 新增: 外部数据整合 =====
    integrator = ExternalDataIntegrator()
    df_processed = integrator.enrich_dataframe(df_processed)
    print(f"外部数据整合完成，新增 {len(integrator.get_external_feature_names())} 个外部特征")
    # ============================
    
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

    # ===== 复合特征构造 =====
    engineer = FeatureEngineer()
    df_processed = engineer.transform(df_processed)
    print(f"复合特征构造完成，新增 {len(engineer.get_new_feature_names())} 个特征")
    # ========================

    # ===== 新增: 时序聚合特征 =====
    temp_engineer = TemporalFeatureEngineer(window_days=90)
    temp_engineer.fit(df_processed)  # 先 fit 计算全局统计量
    df_processed = temp_engineer.transform(df_processed)
    print(f"时序聚合特征构造完成，新增 {len(temp_engineer.get_feature_descriptions())} 个特征")
    # ============================

    # ===== 新增: 特征交互发现 =====
    interaction_discovery = FeatureInteractionDiscovery(max_interactions=20)
    X_before_interaction = df_processed.drop('is_canceled', axis=1, errors='ignore')
    y_for_interaction = df_processed['is_canceled']

    # 只对部分列进行交互发现（限制数量，防止维度爆炸）
    numeric_cols_for_interaction = X_before_interaction.select_dtypes(include=[np.number]).columns[:15].tolist()
    cat_cols_for_interaction = []  # 已编码的类别列不再重复交互

    X_with_interactions = interaction_discovery.discover_interactions(
        X_before_interaction, y_for_interaction,
        numeric_cols=numeric_cols_for_interaction,
        categorical_cols=cat_cols_for_interaction
    )

    # 将交互特征合并回主 DataFrame
    new_inter_cols = [c for c in X_with_interactions.columns if c not in X_before_interaction.columns]
    for col in new_inter_cols:
        df_processed[col] = X_with_interactions[col].values

    print(f"特征交互发现完成，新增 {len(new_inter_cols)} 个交互特征")
    # ============================

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
    
    # 检查 GPU 可用性
    xgb_gpu, lgb_gpu = check_gpu_availability()
    print(f"XGBoost GPU 可用: {xgb_gpu}")
    print(f"LightGBM GPU 可用: {lgb_gpu}")
    
    # 1. 逻辑回归
    print("训练逻辑回归模型...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = lr
    
    # 2. XGBoost（集成学习，GPU加速）
    print("训练 XGBoost 模型...")
    if xgb_gpu:
        xgb_clf = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            tree_method='hist',
            device='cuda',
            random_state=42,
            eval_metric='logloss'
        )
    else:
        xgb_clf = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            tree_method='hist',
            random_state=42,
            eval_metric='logloss'
        )
    xgb_clf.fit(X_train, y_train)
    models['XGBoost'] = xgb_clf
    
    # 3. LightGBM（集成学习，GPU加速）
    print("训练 LightGBM 模型...")
    if lgb_gpu:
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            device='gpu',
            random_state=42,
            verbose=-1
        )
    else:
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
    lgb_clf.fit(X_train, y_train)
    models['LightGBM'] = lgb_clf
    
    # 4. 神经网络
    print("训练神经网络模型...")
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    mlp.fit(X_train, y_train)
    models['Neural Network'] = mlp

    # 5. CatBoost（对类别特征原生支持更好）
    print("训练 CatBoost 模型...")
    try:
        from catboost import CatBoostClassifier
        cat_clf = CatBoostClassifier(
            iterations=200,
            depth=10,
            learning_rate=0.1,
            random_state=42,
            verbose=0,
            # CatBoost 可自动处理类别特征，但这里我们已做编码，所以用默认设置
        )
        cat_clf.fit(X_train, y_train)
        models['CatBoost'] = cat_clf
        print("CatBoost 模型训练完成")
    except ImportError:
        print("警告: catboost 未安装，跳过 CatBoost 模型")
    except Exception as e:
        print(f"CatBoost 训练出错: {e}")

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
    
    # ===== 新增: 自动记录实验到 MLflow Tracker =====
    print("\n记录训练实验...")
    for name, model in models.items():
        try:
            from mlflow_tracker import auto_log_training

            # 提取可序列化的超参数
            try:
                params = model.get_params()
                # 只保留关键参数，过滤掉不可序列化的
                params = {k: v for k, v in params.items()
                          if isinstance(v, (int, float, str, bool, type(None)))}
            except:
                params = {'model_type': type(model).__name__}

            auto_log_training(
                model_name=name,
                params=params,
                metrics=results[name],
                model_object=model,
                feature_names=feature_names
            )
        except Exception as e:
            print(f"  记录 {name} 实验失败: {e}")
    # ============================================

    # 确保目录存在
    if not os.path.exists('models'):
        os.makedirs('models')

    # 保存结果
    joblib.dump(results, 'models/model_results.pkl')

    # 保存模型
    save_models(models, label_encoders, scaler, feature_names)

    # ===== 新增: 设置漂移检测基线 =====
    try:
        from drift_detector import get_drift_monitor
        monitor = get_drift_monitor()
        monitor.set_baseline(df, snapshot_name=f'training_{datetime.now().strftime("%Y%m%d_%H%M")}')
    except Exception as e:
        print(f"  设置漂移基线失败: {e}")
    # ====================================

    print("\n训练完成！")

if __name__ == "__main__":
    main()

