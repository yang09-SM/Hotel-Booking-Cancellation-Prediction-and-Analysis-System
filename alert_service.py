"""
高风险订单预警通知模块
实时监控新创建的预订，自动评估取消风险并触发预警
支持站内消息、邮件、Webhook 多渠道通知
"""

import os
import json
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from threading import Lock
import sqlite3
import pandas as pd

# 全局配置
ALERT_DB_FILE = 'hotel_bookings.db'  # 复用主数据库
ALERT_THRESHOLD_DEFAULT = 0.70  # 默认预警阈值（取消概率）

# 全局预警配置存储
_alert_config = {
    'threshold': ALERT_THRESHOLD_DEFAULT,
    'enabled_channels': ['internal'],  # 默认仅站内消息
    'email_config': {
        'enabled': False,
        'smtp_host': '',
        'smtp_port': 587,
        'sender_email': '',
        'sender_password': '',
        'recipient_emails': []
    },
    'webhook_config': {
        'enabled': False,
        'url': '',
        'method': 'POST',
        'headers': {'Content-Type': 'application/json'}
    }
}

_config_lock = Lock()

class AlertService:
    """预警服务核心类"""

    def __init__(self, db_file=None):
        self.db_file = db_file or ALERT_DB_FILE
        self._ensure_alerts_table()

    def _ensure_alerts_table(self):
        """确保 alerts 表存在"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                booking_id INTEGER,
                alert_type TEXT DEFAULT 'high_risk',
                cancel_probability REAL,
                risk_factors TEXT,
                status TEXT DEFAULT 'pending',
                channel TEXT DEFAULT 'internal',
                notification_sent_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved_at TIMESTAMP,
                resolution_note TEXT
            )
        ''')

        # 创建预警配置表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alert_config (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()

    def evaluate_and_create_alert(self, booking_data, cancel_probability, model_name='XGBoost'):
        """
        评估预订是否需要触发预警
        参数:
            booking_data: 预订数据字典
            cancel_probability: 模型预测的取消概率
            model_name: 使用的模型名称
        返回:
            (should_alert: bool, alert_info: dict or None)
        """
        config = self.get_alert_config()
        threshold = config.get('threshold', ALERT_THRESHOLD_DEFAULT)

        if cancel_probability >= threshold:
            # 提取关键风险因素
            risk_factors = self._extract_risk_factors(booking_data, cancel_probability)

            # 创建预警记录
            alert_id = self._create_alert_record(
                booking_id=booking_data.get('id'),
                cancel_probability=cancel_probability,
                risk_factors=json.dumps(risk_factors, ensure_ascii=False),
                alert_type='high_risk'
            )

            alert_info = {
                'alert_id': alert_id,
                'booking_id': booking_data.get('id'),
                'cancel_probability': cancel_probability,
                'threshold': threshold,
                'risk_factors': risk_factors,
                'model_used': model_name,
                'created_at': datetime.now().isoformat(),
                'status': 'pending'
            }

            # 发送通知
            self._send_notifications(alert_info, config)

            return True, alert_info

        return False, None

    def _extract_risk_factors(self, booking_data, probability):
        """提取导致高风险的关键因素"""
        factors = []

        lead_time = booking_data.get('lead_time', 0)
        if lead_time and int(lead_time) > 150:
            factors.append(f"提前期过长 ({lead_time}天)")

        prev_cancel = booking_data.get('previous_cancellations', 0)
        if prev_cancel and int(prev_cancel) > 0:
            factors.append(f"历史有{prev_cancel}次取消记录")

        is_repeated = booking_data.get('is_repeated_guest', 0)
        if is_repeated == 0:
            factors.append("首次预订的新客户")

        deposit = booking_data.get('deposit_type', '')
        if deposit == 'No Deposit':
            factors.append("无押金预订（可免费取消）")

        adr = booking_data.get('adr', 0)
        if adr and float(adr) < 80:
            factors.append(f"低房价预订 (¥{adr})")

        market_seg = booking_data.get('market_segment', '')
        if market_seg in ['Online TA', 'Offline TA/TO']:
            factors.append(f"通过旅行社/代理商预订 ({market_seg})")

        if not factors:
            factors.append(f"模型综合评估取消概率高达 {probability:.1%}")

        return factors

    def _create_alert_record(self, booking_id, cancel_probability, risk_factors, alert_type='high_risk'):
        """在数据库中创建预警记录"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO alerts (booking_id, cancel_probability, risk_factors, alert_type, status)
            VALUES (?, ?, ?, ?, 'pending')
        ''', (booking_id, cancel_probability, risk_factors, alert_type))

        alert_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return alert_id

    def _send_notifications(self, alert_info, config):
        """根据配置发送多渠道通知"""
        channels = config.get('enabled_channels', ['internal'])

        for channel in channels:
            try:
                if channel == 'internal':
                    self._send_internal_notification(alert_info)
                elif channel == 'email' and config.get('email_config', {}).get('enabled'):
                    self._send_email_notification(alert_info, config['email_config'])
                elif channel == 'webhook' and config.get('webhook_config', {}).get('enabled'):
                    self._send_webhook_notification(alert_info, config['webhook_config'])
            except Exception as e:
                print(f"{channel} 通知发送失败: {e}")

    def _send_internal_notification(self, alert_info):
        """站内消息（已通过 alerts 表记录实现）"""
        pass  # 预警已写入数据库，前端可通过 API 查询

    def _send_email_notification(self, alert_info, email_config):
        """发送邮件通知"""
        msg = MIMEMultipart()
        msg['From'] = email_config['sender_email']
        msg['To'] = ', '.join(email_config['recipient_emails'])
        msg['Subject'] = f"[预警] 高风险订单 #{alert_info['booking_id']} - 取消概率 {alert_info['cancel_probability']:.1%}"

        body = f"""
<h2>酒店预订取消风险预警</h2>
<p><strong>订单编号:</strong> #{alert_info['booking_id']}</p>
<p><strong>取消概率:</strong> <span style="color:red;font-size:18px;font-weight:bold">{alert_info['cancel_probability']:.1%}</span></p>
<p><strong>预警时间:</strong> {alert_info['created_at']}</p>
<p><strong>使用模型:</strong> {alert_info.get('model_used', 'XGBoost')}</p>

<h3>风险因素</h3>
<ul>
"""
        for factor in alert_info.get('risk_factors', []):
            body += f"<li>{factor}</li>\n"

        body += f"""
</ul>

<p style="color:gray;font-size:12px">此邮件由酒店预订智能管理系统自动发送</p>
"""

        msg.attach(MIMEText(body, 'html', 'utf-8'))

        with smtplib.SMTP(email_config['smtp_host'], email_config['smtp_port']) as server:
            server.starttls()
            server.login(email_config['sender_email'], email_config['sender_password'])
            server.send_message(msg)

        # 标记邮件已发送
        self._mark_notification_sent(alert_info['alert_id'], 'email')

    def _send_webhook_notification(self, alert_info, webhook_config):
        """发送 Webhook 通知"""
        payload = {
            'alert_type': 'high_risk_cancellation',
            'severity': 'warning',
            'timestamp': datetime.now().isoformat(),
            'data': alert_info
        }

        response = requests.request(
            method=webhook_config.get('method', 'POST'),
            url=webhook_config['url'],
            json=payload,
            headers=webhook_config.get('headers', {}),
            timeout=10
        )
        response.raise_for_status()

        self._mark_notification_sent(alert_info['alert_id'], 'webhook')

    def _mark_notification_sent(self, alert_id, channel):
        """标记通知已发送"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE alerts SET notification_sent_at = ?, channel = ?
            WHERE id = ?
        ''', (datetime.now().isoformat(), channel, alert_id))
        conn.commit()
        conn.close()

    # ===== 配置管理 =====

    @staticmethod
    def get_alert_config():
        """获取当前预警配置"""
        global _alert_config, _config_lock

        with _config_lock:
            # 尝试从数据库加载
            try:
                conn = sqlite3.connect(ALERT_DB_FILE)
                cursor = conn.cursor()
                cursor.execute('SELECT key, value FROM alert_config')
                rows = cursor.fetchall()
                conn.close()

                db_config = {}
                for key, value in rows:
                    try:
                        db_config[key] = json.loads(value)
                    except:
                        db_config[key] = value

                if db_config:
                    _alert_config.update(db_config)
            except:
                pass

            return _alert_config.copy()

    @staticmethod
    def update_alert_config(new_config):
        """更新预警配置"""
        global _alert_config, _config_lock

        with _config_lock:
            _alert_config.update(new_config)

            # 持久化到数据库
            try:
                conn = sqlite3.connect(ALERT_DB_FILE)
                cursor = conn.cursor()

                for key, value in new_config.items():
                    value_json = json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else str(value)
                    cursor.execute('''
                        INSERT OR REPLACE INTO alert_config (key, value, updated_at)
                        VALUES (?, ?, CURRENT_TIMESTAMP)
                    ''', (key, value_json))

                conn.commit()
                conn.close()
            except Exception as e:
                print(f"配置保存失败: {e}")
                raise

    # ===== 查询接口 =====

    def get_alerts(self, status=None, limit=50, offset=0):
        """获取预警列表"""
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if status:
            cursor.execute('''
                SELECT * FROM alerts WHERE status = ? ORDER BY created_at DESC LIMIT ? OFFSET ?
            ''', (status, limit, offset))
        else:
            cursor.execute('''
                SELECT * FROM alerts ORDER BY created_at DESC LIMIT ? OFFSET ?
            ''', (limit, offset))

        rows = cursor.fetchall()

        cursor.execute('SELECT COUNT(*) FROM alerts')
        total = cursor.fetchone()[0]

        conn.close()

        alerts = []
        for row in rows:
            alert = dict(row)
            if alert.get('risk_factors'):
                try:
                    alert['risk_factors'] = json.loads(alert['risk_factors'])
                except:
                    pass
            alerts.append(alert)

        return alerts, total

    def get_alert_statistics(self):
        """获取预警统计数据"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()

        stats = {}

        # 各状态计数
        cursor.execute('SELECT status, COUNT(*) FROM alerts GROUP BY status')
        status_counts = dict(cursor.fetchall())
        stats['by_status'] = status_counts

        # 总数
        cursor.execute('SELECT COUNT(*) FROM alerts')
        stats['total'] = cursor.fetchone()[0]

        # 今日新增
        today = datetime.now().strftime('%Y-%m-%d')
        cursor.execute("SELECT COUNT(*) FROM alerts WHERE date(created_at) = ?", (today,))
        stats['today_new'] = cursor.fetchone()[0]

        # 近7天趋势
        cursor.execute('''
            SELECT date(created_at) as dt, COUNT(*)
            FROM alerts
            WHERE created_at >= date('now', '-7 days')
            GROUP BY date(created_at)
            ORDER BY dt
        ''')
        stats['weekly_trend'] = [(row[0], row[1]) for row in cursor.fetchall()]

        # 平均响应时间（从创建到解决）
        cursor.execute('''
            SELECT AVG(julianday(resolved_at) - julianday(created_at)) * 24 * 60
            FROM alerts WHERE status = 'resolved' AND resolved_at IS NOT NULL
        ''')
        avg_resolution_minutes = cursor.fetchone()[0]
        stats['avg_resolution_minutes'] = round(avg_resolution_minutes, 1) if avg_resolution_minutes else None

        conn.close()

        return stats

    def resolve_alert(self, alert_id, note=''):
        """解决预警"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE alerts SET status = 'resolved', resolved_at = CURRENT_TIMESTAMP, resolution_note = ?
            WHERE id = ?
        ''', (note, alert_id))
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return success

    def batch_evaluate_bookings(self, bookings_list, prediction_service):
        """
        批量评估预订列表中的高风险订单
        用于对现有数据进行批量扫描
        """
        config = self.get_alert_config()
        threshold = config.get('threshold', ALERT_THRESHOLD_DEFAULT)

        alerts_created = []

        for booking in bookings_list:
            try:
                booking_input = {k: v for k, v in booking.items()
                               if k not in ['id', 'is_canceled', 'reservation_status', 'reservation_status_date']}

                pred = prediction_service.predict(booking_input, 'XGBoost')
                prob = pred.get('probability', {}).get('canceled', 0)

                if prob >= threshold:
                    should_alert, alert_info = self.evaluate_and_create_alert(
                        booking_data=booking,
                        cancel_probability=prob
                    )
                    if should_alert:
                        alerts_created.append(alert_info)
            except Exception as e:
                print(f"评估预订 {booking.get('id')} 失败: {e}")

        return alerts_created
