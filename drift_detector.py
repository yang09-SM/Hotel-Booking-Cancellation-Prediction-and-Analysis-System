"""
概念漂移检测模块
监控模型性能衰减和数据分布变化
使用 PSI（Population Stability Index）监测特征分布偏移
使用 DDM（Drift Detection Method）监测预测误差变化
当检测到显著漂移时触发告警并建议重训练
"""

import os
import json
import numpy as np
from datetime import datetime, timedelta
from collections import deque
import sqlite3

# 配置常量
PSI_WARNING_THRESHOLD = 0.10    # PSI 警告阈值（轻微漂移）
PSI_CRITICAL_THRESHOLD = 0.25   # PSI 严重阈值（显著漂移）
DDM_WARNING_LEVEL = 2          # DDM 警告级别
DDM_CRITICAL_LEVEL = 3         # DDM 严重级别
DRIFT_DB_FILE = 'hotel_bookings.db'
PREDICTION_HISTORY_MAX_SIZE = 1000  # 保留的最近预测记录数


class PSICalculator:
    """Population Stability Index 计算器"""

    @staticmethod
    def calculate(expected_distribution, actual_distribution, bins=10):
        """
        计算 PSI 值

        参数:
            expected_distribution: 参考分布（基线/训练数据）
            actual_distribution: 当前分布（新数据）
            bins: 分箱数量

        返回:
            psi_value: 总体 PSI 值
            bin_details: 各分箱的 PSI 详情
        """
        # 将连续值分箱
        expected_bins = np.histogram(expected_distribution, bins=bins)[0].astype(float)
        actual_bins = np.histogram(actual_distribution, bins=bins)[0].astype(float)

        # 避免除零：给每个箱加一个小常数
        expected_bins = np.where(expected_bins == 0, 0.001, expected_bins)
        actual_bins = np.where(actual_bins == 0, 0.001, actual_bins)

        # 归一化为比例
        expected_pct = expected_bins / expected_bins.sum()
        actual_pct = actual_bins / actual_bins.sum()

        # 计算 PSI = sum((actual - expected) * ln(actual/expected))
        psi_per_bin = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
        psi_total = np.sum(psi_per_bin)

        bin_details = [
            {
                'bin_index': i,
                'expected_pct': round(float(expected_pct[i]), 4),
                'actual_pct': round(float(actual_pct[i]), 4),
                'psi': round(float(psi_per_bin[i]), 4)
            }
            for i in range(len(psi_per_bin))
        ]

        return round(float(psi_total), 4), bin_details

    @staticmethod
    def interpret_psi(psi_value):
        """解释 PSI 值的含义"""
        if psi_value < 0.05:
            return 'stable', '分布稳定，无明显漂移'
        elif psi_value < 0.10:
            return 'slight', '轻微漂移，建议关注'
        elif psi_value < 0.25:
            return 'moderate', '中等漂移，建议检查数据'
        else:
            return 'severe', '严重漂移，建议立即重训练'


class DDMDetector:
    """
    Drift Detection Method 检测器
    基于 EWM（Exponential Weighted Mean）和标准差的变化检测误差漂移
    参考: Gama et al., "Learning with Drift Detection" (2004)
    """

    def __init__(self, warning_level=2, critical_level=3, min_num_instances=30):
        self.warning_level = warning_level
        self.critical_level = critical_level
        self.min_num_instances = min_num_instances

        self.n_instances = 0
        self.ewm_mean = None   # Exponential Weighted Mean
        self.ewm_std = None    # Exponential Weighted Standard Deviation
        self.alpha = 0.05      # 平滑因子

        self.in_warning_zone = False
        self.drift_detected = False
        self.history = []      # 保留历史用于分析

    def add_prediction(self, predicted_prob, actual_label=None):
        """
        添加一个预测结果用于监控

        参数:
            predicted_prob: 模型预测的概率值
            actual_label: 实际标签（如果已知，用于计算误差）

        返回:
            status: 'normal' | 'warning' | 'drift' | 'insufficient_data'
        """
        self.n_instances += 1

        if actual_label is not None:
            error = abs(predicted_prob - actual_label)
        else:
            # 无真实标签时使用概率值的波动作为代理指标
            error = predicted_prob * (1 - predicted_prob)  # 不确定性度量

        self.history.append({
            'predicted_prob': predicted_prob,
            'error': error,
            'timestamp': datetime.now().isoformat(),
            'has_ground_truth': actual_label is not None
        })

        # 保持历史大小限制
        if len(self.history) > PREDICTION_HISTORY_MAX_SIZE:
            self.history = self.history[-PREDICTION_HISTORY_MAX_SIZE:]

        if self.n_instances < self.min_num_instances:
            return 'insufficient_data'

        # 初始化 EWM
        if self.ewm_mean is None:
            self.ewm_mean = error
            self.ewm_std = 0.0
            return 'normal'

        # 更新 EWM
        self.ewm_mean = self.alpha * error + (1 - self.alpha) * self.ewm_mean
        self.ewm_std = np.sqrt(
            self.alpha * (error - self.ewm_mean)**2 + (1 - self.alpha) * self.ewm_std**2
        )

        # 检测漂移
        current_level = (error - self.ewm_mean) / (self.ewm_std + 1e-10)

        if current_level > self.critical_level:
            self.drift_detected = True
            self._reset()
            return 'drift'
        elif current_level > self.warning_level:
            self.in_warning_zone = True
            return 'warning'
        else:
            if self.in_warning_zone:
                self.in_warning_zone = False
            return 'normal'

    def _reset(self):
        """检测到漂移后重置状态"""
        self.n_instances = 0
        self.ewm_mean = None
        self.ewm_std = None
        self.in_warning_zone = False

    def get_status(self):
        """获取当前检测器状态"""
        return {
            'n_instances': self.n_instances,
            'ewm_mean': round(self.ewm_mean, 6) if self.ewm_mean is not None else None,
            'ewm_std': round(self.ewm_std, 6) if self.ewm_std is not None else None,
            'in_warning_zone': self.in_warning_zone,
            'drift_detected': self.drift_detected,
            'history_size': len(self.history)
        }


class ConceptDriftMonitor:
    """概念漂移综合监控系统"""

    def __init__(self, db_file=None):
        self.db_file = db_file or DRIFT_DB_FILE
        self.psi_calculator = PSICalculator()
        self.ddm_detector = DDMDetector()
        self.baseline_features = {}     # 基线特征分布
        self._ensure_monitoring_table()
        self._load_baseline_if_exists()

    def _ensure_monitoring_table(self):
        """确保漂移监控相关表存在"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS drift_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT,
                feature_name TEXT,
                psi_value REAL,
                ddm_status TEXT,
                severity TEXT,
                details TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS baseline_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_name TEXT,
                feature_stats TEXT,
                model_version TEXT,
                data_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()

    def _load_baseline_if_exists(self):
        """加载已有的基线快照"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT feature_stats FROM baseline_snapshots
                ORDER BY created_at DESC LIMIT 1
            ''')
            row = cursor.fetchone()
            if row and row[0]:
                self.baseline_features = json.loads(row[0])
            conn.close()
        except:
            pass

    def set_baseline(self, df, snapshot_name='auto', model_version='unknown'):
        """
        设置基线分布（通常在模型训练后调用）
        从训练数据中提取各特征的统计分布作为基线
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        baseline_stats = {}
        for col in numeric_cols:
            values = df[col].dropna().values
            if len(values) > 0:
                baseline_stats[col] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values)),
                    'q25': float(np.percentile(values, 25)),
                    'q75': float(np.percentile(values, 75)),
                    'histogram': np.histogram(values, bins=20)[0].tolist(),
                    'count': int(len(values))
                }

        self.baseline_features = baseline_stats

        # 持久化到数据库
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO baseline_snapshots (snapshot_name, feature_stats, model_version, data_count)
            VALUES (?, ?, ?, ?)
        ''', (snapshot_name, json.dumps(baseline_stats, ensure_ascii=False),
              model_version, len(df)))
        conn.commit()
        conn.close()

        print(f"[Drift Monitor] 基线已设置，包含 {len(baseline_stats)} 个数值特征")
        return baseline_stats

    def check_feature_drift(self, current_df):
        """
        检查当前数据的特征分布是否相对于基线发生了漂移
        返回所有特征的 PSI 分析结果
        """
        results = {}
        max_psi = 0
        drifted_features = []

        for col, baseline in self.baseline_features.items():
            if col not in current_df.columns:
                continue

            current_values = current_df[col].dropna().values

            if len(current_values) < 50 or 'histogram' not in baseline:
                continue

            # 使用基线的分箱边界对当前数据进行分箱
            psi_value, bin_details = self.psi_calculator.calculate(
                expected_distribution=np.random.choice(
                    current_values,
                    size=min(len(current_values), baseline.get('count', 1000)),
                    replace=True
                ) if len(current_values) < 100 else current_values,
                actual_distribution=current_values,
                bins=10
            )

            level, interpretation = self.psi_calculator.interpret_psi(psi_value)

            results[col] = {
                'psi': psi_value,
                'level': level,
                'interpretation': interpretation,
                'bin_details': bin_details[:5]  # 只返回前5个箱
            }

            if psi_value > max_psi:
                max_psi = psi_value

            if level in ['moderate', 'severe']:
                drifted_features.append(col)

        # 记录严重漂移事件
        for col in drifted_features:
            if results[col]['level'] == 'severe':
                self._record_drift_event(
                    event_type='psi_critical',
                    feature_name=col,
                    psi_value=results[col]['psi'],
                    severity='critical',
                    details=json.dumps(results[col], ensure_ascii=False)
                )

        overall_status = 'stable' if max_psi < PSI_WARNING_THRESHOLD else \
                        'warning' if max_psi < PSI_CRITICAL_THRESHOLD else 'critical'

        return {
            'overall_status': overall_status,
            'max_psi': max_psi,
            'features_analyzed': len(results),
            'drifted_features': drifted_features,
            'feature_details': results
        }

    def check_model_performance_drift(self, recent_predictions):
        """
        监控模型预测性能是否出现漂移
        通过 DDM 方法检测误差模式变化

        参数:
            recent_predictions: 最近一批预测结果列表
                格式: [{'predicted_prob': float, 'actual_label': int/None}, ...]
        """
        ddm_results = []

        for pred in recent_predictions:
            prob = pred.get('predicted_prob', pred.get('probability', {}).get('canceled', 0.5))
            actual = pred.get('actual_label')

            status = self.ddm_detector.add_prediction(prob, actual)
            ddm_results.append(status)

        ddm_status = self.ddm_detector.get_status()

        if ddm_status['drift_detected']:
            self._record_drift_event(
                event_type='ddm_drift',
                feature_name='model_error',
                ddm_status='drift',
                severity='critical',
                details=json.dumps(ddm_status, ensure_ascii=False)
            )

        return {
            'ddm_status': ddm_status['drift_detected'],
            'in_warning_zone': ddm_status['in_warning_zone'],
            'recent_predictions_count': len(recent_predictions),
            'status_breakdown': {s: ddm_results.count(s) for s in set(ddm_results)}
        }

    def _record_drift_event(self, event_type, feature_name, psi_value=None,
                            ddm_status=None, severity='warning', details=None):
        """记录漂移事件"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO drift_events (event_type, feature_name, psi_value, ddm_status, severity, details)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (event_type, feature_name, psi_value, ddm_status, severity, details))
        conn.commit()
        conn.close()

    def get_drift_history(self, limit=50, event_type=None):
        """获取漂移事件历史"""
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if event_type:
            cursor.execute('''
                SELECT * FROM drift_events WHERE event_type = ?
                ORDER BY created_at DESC LIMIT ?
            ''', (event_type, limit))
        else:
            cursor.execute('SELECT * FROM drift_events ORDER BY created_at DESC LIMIT ?', (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_comprehensive_report(self):
        """生成综合漂移检测报告"""
        ddm_status = self.ddm_detector.get_status()

        # 获取最近的漂移事件
        recent_events = self.get_drift_history(limit=10)

        # 统计各类事件数量
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute('SELECT event_type, COUNT(*) FROM drift_events GROUP BY event_type')
        event_counts = dict(cursor.fetchall())
        conn.close()

        return {
            'monitoring_status': {
                'baseline_set': len(self.baseline_features) > 0,
                'baseline_features_count': len(self.baseline_features),
                'ddm_instances_tracked': ddm_status['n_instances'],
                'ddm_in_warning_zone': ddm_status['in_warning_zone'],
                'ddm_last_drift': ddm_status['drift_detected']
            },
            'thresholds': {
                'psi_warning': PSI_WARNING_THRESHOLD,
                'psi_critical': PSI_CRITICAL_THRESHOLD,
                'ddm_warning_level': DDM_WARNING_LEVEL,
                'ddm_critical_level': DDM_CRITICAL_LEVEL
            },
            'recent_events': recent_events[:5],
            'event_summary': event_counts,
            'recommendations': self._generate_recommendations(ddm_status, event_counts)
        }

    def _generate_recommendations(self, ddm_status, event_counts):
        """根据当前状态生成建议"""
        recommendations = []

        total_events = sum(event_counts.values())
        recent_critical = event_counts.get('psi_critical', 0) + event_counts.get('ddm_drift', 0)

        if ddm_status['drift_detected']:
            recommendations.append({
                'priority': 'high',
                'action': 'immediate_retrain',
                'message': '检测到模型性能漂移，建议立即启动模型重训练流程'
            })

        if recent_critical >= 3:
            recommendations.append({
                'priority': 'high',
                'action': 'investigate_data_source',
                'message': f'近期有 {recent_critical} 次严重漂移事件，请检查数据源是否发生变化'
            })

        if ddm_status['in_warning_zone']:
            recommendations.append({
                'priority': 'medium',
                'action': 'monitor_closely',
                'message': '模型处于警告区域，请密切监控后续预测表现'
            })

        if total_events == 0:
            recommendations.append({
                'priority': 'info',
                'action': 'none',
                'message': '系统运行正常，未检测到显著漂移'
            })

        return recommendations


# 全局监控实例
_monitor_instance = None

def get_drift_monitor():
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = ConceptDriftMonitor()
    return _monitor_instance


def auto_check_and_alert():
    """自动执行漂移检测并发送告警（可由定时任务调用）"""
    monitor = get_drift_monitor()

    if not monitor.baseline_features:
        return {'status': 'no_baseline', 'message': '尚未设置基线，请在训练后设置'}

    # 加载最近的数据进行检测
    import pandas as pd
    import database

    try:
        conn = sqlite3.connect(monitor.db_file)
        # 取最近 500 条预订数据进行对比
        df = pd.read_sql_query('SELECT * FROM bookings ORDER BY id DESC LIMIT 500', conn)
        conn.close()

        if len(df) < 50:
            return {'status': 'insufficient_data', 'message': '数据量不足'}

        psi_result = monitor.check_feature_drift(df)
        report = monitor.get_comprehensive_report()

        return {
            'psi_result': psi_result,
            'report': report
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}
