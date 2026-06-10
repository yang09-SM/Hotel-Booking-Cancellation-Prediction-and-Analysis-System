"""
超参数自动优化模块
基于 Optuna 框架实现贝叶斯超参数搜索
支持早停机制和交叉验证
"""

import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import joblib
import os
import json
import time
from datetime import datetime

# 全局优化状态存储（用于异步查询进度）
_optimization_status = {
    'is_running': False,
    'current_trial': 0,
    'total_trials': 0,
    'best_score': 0,
    'model_name': '',
    'start_time': None,
    'result': None,
    'error': None
}

class HyperparameterOptimizer:
    """超参数优化器"""

    def __init__(self, models_dir='models', data_file='hotel_bookings.csv'):
        self.models_dir = models_dir
        self.data_file = data_file
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.label_encoders = None
        self.scaler = None

    def load_and_preprocess_data(self):
        """加载并预处理数据（复用 train_models.py 的逻辑）"""
        from train_models import load_data, preprocess_data

        df = load_data(self.data_file)
        X, y, label_encoders, scaler, categorical_cols, numeric_cols = preprocess_data(df)

        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.feature_names = X.columns.tolist()
        self.label_encoders = label_encoders
        self.scaler = scaler

    def optimize_xgboost(self, n_trials=50, timeout=600):
        """
        优化 XGBoost 超参数
        使用 TPE 采样器 + 早停策略
        """
        import xgboost as xgb

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': 42,
                'eval_metric': 'logloss',
                'tree_method': 'hist',
                'verbosity': 0
            }

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = []

            for train_idx, val_idx in cv.split(self.X_train, self.y_train):
                X_tr, X_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
                y_tr, y_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]

                model = xgb.XGBClassifier(**params)
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

                y_pred_proba = model.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, y_pred_proba)
                cv_scores.append(score)

            return np.mean(cv_scores)

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

        return study

    def optimize_lightgbm(self, n_trials=50, timeout=600):
        """优化 LightGBM 超参数"""
        import lightgbm as lgb

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': 42,
                'verbose': -1
            }

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = []

            for train_idx, val_idx in cv.split(self.X_train, self.y_train):
                X_tr, X_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
                y_tr, y_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]

                model = lgb.LGBMClassifier(**params)
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])

                y_pred_proba = model.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, y_pred_proba)
                cv_scores.append(score)

            return np.mean(cv_scores)

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

        return study

    def optimize_mlp(self, n_trials=30, timeout=300):
        """优化神经网络(MLP)超参数"""
        from sklearn.neural_network import MLPClassifier

        def objective(trial):
            params = {
                'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [
                    (64,), (128,), (64, 32), (128, 64), (100, 50), (200, 100, 50)
                ]),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
                'alpha': trial.suggest_float('alpha', 1e-5, 0.1, log=True),
                'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 0.01, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
                'max_iter': trial.suggest_int('max_iter', 200, 800),
                'random_state': 42
            }

            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # MLP较慢，用3折
            cv_scores = []

            for train_idx, val_idx in cv.split(self.X_train, self.y_train):
                X_tr, X_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
                y_tr, y_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]

                model = MLPClassifier(**params)
                model.fit(X_tr, y_tr)

                y_pred_proba = model.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, y_pred_proba)
                cv_scores.append(score)

            return np.mean(cv_scores)

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

        return study

    def train_best_model(self, study, model_name):
        """使用最优超参数训练最终模型并在测试集上评估"""
        best_params = study.best_params.copy()
        best_params['random_state'] = 42

        if model_name == 'XGBoost':
            import xgboost as xgb
            # 移除非构造函数参数
            best_params.pop('n_estimators', None)  # XGBClassifier 用 n_estimators
            model = xgb.XGBClassifier(**best_params, n_estimators=study.best_params.get('n_estimators', 200),
                                       eval_metric='logloss', tree_method='hist')
        elif model_name == 'LightGBM':
            import lightgbm as lgb
            best_params.pop('n_estimators', None)
            model = lgb.LGBMClassifier(**best_params, n_estimators=study.best_params.get('n_estimators', 200), verbose=-1)
        elif model_name == 'Neural Network':
            from sklearn.neural_network import MLPClassifier
            model = MLPClassifier(**best_params)
        else:
            raise ValueError(f"不支持的模型: {model_name}")

        model.fit(self.X_train, self.y_train)

        # 测试集评估
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

        test_metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba)
        }

        return model, test_metrics

    def save_optimization_result(self, study, model, metrics, model_name):
        """保存优化结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存最优模型
        safe_name = model_name.replace(' ', '_')
        joblib.dump(model, f'{self.models_dir}/{safe_name}_optimized_model.pkl')

        # 保存实验日志
        experiment_log = {
            'model_name': model_name,
            'timestamp': timestamp,
            'best_params': study.best_params,
            'best_cv_score': float(study.best_value),
            'n_trials': len(study.trials),
            'test_metrics': {k: float(v) for k, v in metrics.items()},
            'optimization_history': [
                {'trial_number': t.number, 'value': float(t.value) if t.value else None,
                 'params': t.params}
                for t in study.trials if t.value is not None
            ]
        }

        log_path = f'{self.models_dir}/optimization_{safe_name}_{timestamp}.json'
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_log, f, ensure_ascii=False, indent=2)

        return log_path


def start_optimization_async(model_name, n_trials=50, timeout=600):
    """启动异步优化任务"""
    global _optimization_status

    _optimization_status.update({
        'is_running': True,
        'current_trial': 0,
        'total_trials': n_trials,
        'best_score': 0,
        'model_name': model_name,
        'start_time': time.time(),
        'result': None,
        'error': None
    })

    def run_optimization():
        global _optimization_status
        try:
            optimizer = HyperparameterOptimizer()
            optimizer.load_and_preprocess_data()

            # 根据模型名选择优化方法
            if model_name == 'XGBoost':
                study = optimizer.optimize_xgboost(n_trials, timeout)
            elif model_name == 'LightGBM':
                study = optimizer.optimize_lightgbm(n_trials, timeout)
            elif model_name in ['Neural Network', 'MLP']:
                study = optimizer.optimize_mlp(n_trials, timeout)
            else:
                raise ValueError(f"不支持优化的模型: {model_name}")

            # 训练最优模型
            model, metrics = optimizer.train_best_model(study, model_name)

            # 保存结果
            log_path = optimizer.save_optimization_result(study, model, metrics, model_name)

            _optimization_status.update({
                'is_running': False,
                'best_score': float(study.best_value),
                'result': {
                    'model_name': model_name,
                    'best_params': study.best_params,
                    'best_cv_score': float(study.best_value),
                    'test_metrics': {k: float(v) for k, v in metrics.items()},
                    'log_path': log_path,
                    'n_trials': len(study.trials)
                }
            })
        except Exception as e:
            _optimization_status.update({
                'is_running': False,
                'error': str(e)
            })

    # 在后台线程中运行
    import threading
    thread = threading.Thread(target=run_optimization, daemon=True)
    thread.start()

def get_optimization_status():
    """获取当前优化状态"""
    global _optimization_status

    status = _optimization_status.copy()
    if status['is_running'] and status['start_time']:
        elapsed = time.time() - status['start_time']
        status['elapsed_seconds'] = round(elapsed, 1)

    return status
