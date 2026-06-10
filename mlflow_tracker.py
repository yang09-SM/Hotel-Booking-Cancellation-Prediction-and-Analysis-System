"""
MLflow 实验追踪与模型版本管理模块
记录每次模型训练的超参数、指标、artifact 和代码版本
支持实验对比和模型回滚
"""

import os
import json
import joblib
import numpy as np
from datetime import datetime

class MLflowTracker:
    """简化的 MLflow 风格实验追踪器（不依赖 MLflow 库，使用本地文件系统）"""

    def __init__(self, experiments_dir='mlflow_experiments'):
        self.experiments_dir = experiments_dir
        self.experiments_file = os.path.join(experiments_dir, 'experiments.json')
        self._ensure_dir()

    def _ensure_dir(self):
        if not os.path.exists(self.experiments_dir):
            os.makedirs(self.experiments_dir)

        if not os.path.exists(self.experiments_file):
            with open(self.experiments_file, 'w', encoding='utf-8') as f:
                json.dump({'experiments': []}, f, ensure_ascii=False)

    def log_experiment(
        self,
        experiment_name,
        model_name,
        params,
        metrics,
        model_artifact=None,
        tags=None,
        feature_names=None
    ):
        """
        记录一次训练实验

        参数:
            experiment_name: 实验名称（如 'XGBoost_Optimized_v2'）
            model_name: 模型名称（如 'XGBoost'）
            params: 超参数字典
            metrics: 评估指标字典 {accuracy, precision, recall, f1, roc_auc}
            model_artifact: 模型对象（可选，用于保存 artifact）
            tags: 标签字典（可选）
            feature_names: 特征名称列表（可选）

        返回:
            experiment_id (str)
        """
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_name}"

        # 构建实验记录
        experiment_record = {
            'experiment_id': experiment_id,
            'experiment_name': experiment_name,
            'model_name': model_name,
            'params': params,
            'metrics': {k: float(v) for k, v in metrics.items()},
            'tags': tags or {},
            'feature_count': len(feature_names) if feature_names else None,
            'created_at': datetime.now().isoformat(),
            'status': 'completed',
            'artifact_path': None
        }

        # 保存模型 artifact
        if model_artifact is not None:
            artifact_dir = os.path.join(self.experiments_dir, experiment_id)
            os.makedirs(artifact_dir, exist_ok=True)
            artifact_path = os.path.join(artifact_dir, 'model.pkl')
            joblib.dump(model_artifact, artifact_path)

            # 同时保存特征名称
            if feature_names:
                with open(os.path.join(artifact_dir, 'feature_names.json'), 'w') as f:
                    json.dump(feature_names, f)

            experiment_record['artifact_path'] = artifact_path

        # 追加到实验列表
        with open(self.experiments_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        data['experiments'].append(experiment_record)

        # 保持最多100条记录
        if len(data['experiments']) > 100:
            data['experiments'] = data['experiments'][-100:]

        with open(self.experiments_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"[MLflow Tracker] 实验已记录: {experiment_id}")
        print(f"  模型: {model_name} | AUC: {metrics.get('roc_auc', 'N/A'):.4f} | F1: {metrics.get('f1', 'N/A'):.4f}")

        return experiment_id

    def get_all_experiments(self, model_name=None, limit=50):
        """获取所有实验记录"""
        with open(self.experiments_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        experiments = data.get('experiments', [])

        if model_name:
            experiments = [e for e in experiments if e.get('model_name') == model_name]

        # 按时间倒序
        experiments.sort(key=lambda x: x.get('created_at', ''), reverse=True)

        return experiments[:limit], len(experiments)

    def get_experiment(self, experiment_id):
        """获取单个实验详情"""
        experiments, _ = self.get_all_experiments()
        for exp in experiments:
            if exp.get('experiment_id') == experiment_id:
                return exp
        return None

    def get_best_experiment(self, model_name, metric='roc_auc'):
        """获取指定模型的最佳实验（按指定指标）"""
        experiments, _ = self.get_all_experiments(model_name=model_name)

        if not experiments:
            return None

        best = max(
            [e for e in experiments if metric in e.get('metrics', {})],
            key=lambda x: x['metrics'][metric]
        )
        return best

    def compare_experiments(self, experiment_ids):
        """对比多个实验"""
        results = []
        for exp_id in experiment_ids:
            exp = self.get_experiment(exp_id)
            if exp:
                results.append(exp)
        return results

    def delete_experiment(self, experiment_id):
        """删除实验记录及其 artifact"""
        with open(self.experiments_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        original_len = len(data['experiments'])
        data['experiments'] = [e for e in data['experiments']
                               if e.get('experiment_id') != experiment_id]

        with open(self.experiments_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # 删除 artifact 目录
        artifact_dir = os.path.join(self.experiments_dir, experiment_id)
        if os.path.exists(artifact_dir):
            import shutil
            shutil.rmtree(artifact_dir)

        return len(data['experiments']) < original_len


# 全局 tracker 实例
_tracker_instance = None

def get_mlflow_tracker():
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = MLflowTracker()
    return _tracker_instance


def auto_log_training(model_name, params, metrics, model_object, feature_names=None):
    """便捷函数：自动记录一次训练"""
    tracker = get_mlflow_tracker()

    experiment_name = f"{model_name}_training"
    if any(k.startswith('optuna') or k.startswith('n_estimators') for k in params.keys()):
        experiment_name = f"{model_name}_optimized"

    return tracker.log_experiment(
        experiment_name=experiment_name,
        model_name=model_name,
        params=params,
        metrics=metrics,
        model_artifact=model_object,
        tags={'auto_logged': 'true'},
        feature_names=feature_names
    )
