
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import database
from prediction_service import get_prediction_service
# 导入认证与权限控制模块
from auth import login_required, role_required, authenticate_user
# 导入缓存服务
from cache_service import cached, invalidate_cache, get_cache_stats
import os
import webbrowser
import threading

app = Flask(__name__)
CORS(app)

# 前端页面路由
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# 初始化数据库
if not os.path.exists('hotel_bookings.db'):
    database.init_db()
    if os.path.exists('hotel_bookings.csv'):
        database.import_csv_to_db('hotel_bookings.csv')

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'message': 'Hotel Booking System is running'})

# ==================== 认证相关API ====================

# 用户登录接口（公开路由，无需认证）
@app.route('/api/auth/login', methods=['POST'])
def login():
    """用户登录接口"""
    data = request.json
    username = data.get('username')
    password = data.get('password')

    result = authenticate_user(username, password)

    # authenticate_user 返回 (success, message, user_info) 元组
    if result[0]:  # success
        user_info = result[2]  # user_info
        return jsonify({
            'token': user_info['token'],
            'user': {
                'id': user_info['id'],
                'username': user_info['username'],
                'role': user_info['role']
            }
        })
    return jsonify({'error': result[1]}), 401  # 返回错误消息

# 获取当前登录用户信息
@app.route('/api/auth/me', methods=['GET'])
@login_required
def get_current_user_info():
    """获取当前登录用户信息"""
    user = request.current_user
    return jsonify({
        'id': user['user_id'],
        'username': user['username'],
        'role': user['role']
    })

# ==================== 预订管理API ====================

# 获取预订列表 - 店员及以上可访问
@app.route('/api/bookings', methods=['GET'])
@login_required
@role_required(['staff', 'manager', 'admin'])
def get_bookings():
    limit = request.args.get('limit', 100, type=int)
    offset = request.args.get('offset', 0, type=int)
    keyword = request.args.get('keyword', None)
    
    if keyword:
        bookings = database.search_bookings(keyword, limit)
        total = len(bookings)
    else:
        bookings, total = database.get_all_bookings(limit, offset)
    
    return jsonify({
        'bookings': bookings,
        'total': total,
        'limit': limit,
        'offset': offset
    })

# 获取预订详情 - 店员及以上可访问
@app.route('/api/bookings/<int:booking_id>', methods=['GET'])
@login_required
@role_required(['staff', 'manager', 'admin'])
def get_booking(booking_id):
    booking = database.get_booking_by_id(booking_id)
    if booking:
        return jsonify(booking)
    return jsonify({'error': 'Booking not found'}), 404

# 创建预订 - 仅管理员可访问
@app.route('/api/bookings', methods=['POST'])
@login_required
@role_required(['admin'])
def create_booking():
    booking_data = request.json
    booking_id = database.create_booking(booking_data)

    # ===== 新增: 自动预警检查 =====
    try:
        from alert_service import AlertService
        from prediction_service import get_prediction_service

        pred_service = get_prediction_service()
        pred_result = pred_service.predict(booking_data, 'XGBoost')
        cancel_prob = pred_result.get('probability', {}).get('canceled', 0)

        alert_service = AlertService()
        should_alert, alert_info = alert_service.evaluate_and_create_alert(
            booking_data={**booking_data, 'id': booking_id},
            cancel_probability=cancel_prob
        )

        if should_alert:
            print(f"[预警] 新建订单 #{booking_id} 触发高风险预警 (概率: {cancel_prob:.1%})")
    except Exception as e:
        print(f"预警检查失败（不影响预订创建）: {e}")
    # ==================================

    # 清除相关缓存（统计数据变化了）
    try:
        invalidate_cache('hotel:stats:*')  # 统计数据变化了
        invalidate_cache('hotel:bookings:*')  # 预订列表变化了
    except Exception as e:
        print(f"缓存清除失败（不影响预订创建）: {e}")

    return jsonify({'id': booking_id, 'message': 'Booking created successfully'}), 201

# 更新预订 - 仅管理员可访问
@app.route('/api/bookings/<int:booking_id>', methods=['PUT'])
@login_required
@role_required(['admin'])
def update_booking(booking_id):
    booking_data = request.json
    success = database.update_booking(booking_id, booking_data)
    if success:
        # 清除相关缓存（统计数据变化了）
        try:
            invalidate_cache('hotel:stats:*')  # 统计数据变化了
            invalidate_cache('hotel:bookings:*')  # 预订列表变化了
        except Exception as e:
            print(f"缓存清除失败（不影响预订更新）: {e}")

        return jsonify({'message': 'Booking updated successfully'})
    return jsonify({'error': 'Booking not found'}), 404

# 删除预订 - 仅管理员可访问
@app.route('/api/bookings/<int:booking_id>', methods=['DELETE'])
@login_required
@role_required(['admin'])
def delete_booking(booking_id):
    success = database.delete_booking(booking_id)
    if success:
        # 清除相关缓存（统计数据变化了）
        try:
            invalidate_cache('hotel:stats:*')  # 统计数据变化了
            invalidate_cache('hotel:bookings:*')  # 预订列表变化了
        except Exception as e:
            print(f"缓存清除失败（不影响预订删除）: {e}")

        return jsonify({'message': 'Booking deleted successfully'})
    return jsonify({'error': 'Booking not found'}), 404

# ==================== 统计API ====================

# 获取统计信息 - 店员及以上可访问
@app.route('/api/statistics', methods=['GET'])
@login_required
@role_required(['staff', 'manager', 'admin'])
@cached(ttl=60, prefix='stats')
def get_statistics():
    stats = database.get_statistics()
    return jsonify(stats)

# ==================== 预测API ====================

# 执行预测 - 经理及以上可访问
@app.route('/api/predict', methods=['POST'])
@login_required
@role_required(['manager', 'admin'])
def predict():
    from cache_service import get_cache_backend
    import hashlib
    import json

    data = request.json
    booking_data = data.get('booking', {})
    model_name = data.get('model', 'XGBoost')

    # 手动构建缓存键
    cache_key = f"hotel:predict:{hashlib.md5(json.dumps({**booking_data, 'model': model_name}, sort_keys=True).encode()).hexdigest()}"

    cache = get_cache_backend()
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        return jsonify(cached_result)

    try:
        service = get_prediction_service()
        result = service.predict(booking_data, model_name)

        # 写入缓存
        cache.set(cache_key, result, ttl=600)

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 全模型预测 - 经理及以上可访问
@app.route('/api/predict/all', methods=['POST'])
@login_required
@role_required(['manager', 'admin'])
def predict_all():
    data = request.json
    booking_data = data.get('booking', {})

    try:
        service = get_prediction_service()
        results = service.predict_all_models(booking_data)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 融合模型预测 - 经理及以上可访问
@app.route('/api/predict/ensemble', methods=['POST'])
@login_required
@role_required(['manager', 'admin'])
def predict_ensemble():
    """使用融合模型进行预测（Voting 或 Stacking）"""
    data = request.json
    booking_data = data.get('booking', {})
    ensemble_type = data.get('ensemble_type', 'voting')  # 'voting' 或 'stacking'

    if not booking_data:
        return jsonify({'error': '请提供预订数据(booking)'}), 400

    if ensemble_type not in ['voting', 'stacking']:
        return jsonify({'error': '不支持的融合类型，请使用 voting 或 stacking'}), 400

    try:
        service = get_prediction_service()
        result = service.predict_ensemble(booking_data, ensemble_type)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 获取可用模型列表 - 经理及以上可访问
@app.route('/api/models', methods=['GET'])
@login_required
@role_required(['manager', 'admin'])
@cached(ttl=300, prefix='models')
def get_models():
    try:
        service = get_prediction_service()
        models = service.get_available_models()
        return jsonify({'models': models})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 获取模型性能 - 经理及以上可访问
@app.route('/api/models/performance', methods=['GET'])
@login_required
@role_required(['manager', 'admin'])
@cached(ttl=300, prefix='perf')
def get_model_performance():
    try:
        service = get_prediction_service()
        performance = service.get_model_performance()
        if performance:
            return jsonify(performance)
        return jsonify({'message': 'No performance data available'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== SHAP可解释性分析API ====================

# 获取全局特征重要性（SHAP Summary Plot数据）
@app.route('/api/shap/feature-importance', methods=['GET'])
@login_required
@role_required(['manager', 'admin'])
def get_shap_feature_importance():
    """获取全局特征重要性分析结果"""
    model_name = request.args.get('model', 'XGBoost')

    try:
        service = get_prediction_service()
        result = service.get_global_feature_importance(model_name)

        if 'error' in result:
            return jsonify(result), 400

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'获取特征重要性失败: {str(e)}'}), 500

# 单样本预测解释
@app.route('/api/shap/explain', methods=['POST'])
@login_required
@role_required(['manager', 'admin'])
def explain_prediction():
    """获取单样本预测的SHAP详细解释"""
    data = request.json
    booking_data = data.get('booking', {})
    model_name = data.get('model', 'XGBoost')

    if not booking_data:
        return jsonify({'error': '请提供预订数据(booking)'}), 400

    try:
        service = get_prediction_service()
        result = service.get_shap_explanation(booking_data, model_name)

        if 'error' in result:
            return jsonify(result), 400

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'生成预测解释失败: {str(e)}'}), 500

# 获取依赖关系图数据
@app.route('/api/shap/dependence', methods=['GET'])
@login_required
@role_required(['manager', 'admin'])
def get_shap_dependence():
    """获取特征依赖关系图数据"""
    feature = request.args.get('feature')
    model_name = request.args.get('model', 'XGBoost')

    if not feature:
        return jsonify({'error': '请提供特征名称(feature)参数'}), 400

    try:
        service = get_prediction_service()
        result = service.get_dependence_plot_data(feature, model_name)

        if 'error' in result:
            return jsonify(result), 400

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'获取依赖关系数据失败: {str(e)}'}), 500

# 生成SHAP Summary Plot图表（Base64图片）
@app.route('/api/shap/plots/summary', methods=['GET'])
@login_required
@role_required(['manager', 'admin'])
def get_shap_summary_plot():
    """生成SHAP Summary Plot的Base64编码图片"""
    from shap_analyzer import get_shap_analyzer

    model_name = request.args.get('model', 'XGBoost')
    max_display = request.args.get('max_display', 20, type=int)

    try:
        analyzer = get_shap_analyzer()
        img_base64 = analyzer.generate_summary_plot_base64(model_name, max_display)

        if img_base64 is None:
            return jsonify({'error': '无法生成Summary Plot图片'}), 500

        return jsonify({
            'image': img_base64,
            'format': 'png',
            'model': model_name
        })
    except Exception as e:
        return jsonify({'error': f'生成Summary Plot失败: {str(e)}'}), 500

# 生成SHAP Force Plot图表（Base64图片）
@app.route('/api/shap/plots/force', methods=['POST'])
@login_required
@role_required(['manager', 'admin'])
def get_shap_force_plot():
    """生成SHAP Force Plot的Base64编码图片"""
    from shap_analyzer import get_shap_analyzer

    data = request.json
    booking_data = data.get('booking', {})
    model_name = data.get('model', 'XGBoost')

    if not booking_data:
        return jsonify({'error': '请提供预订数据(booking)'}), 400

    try:
        analyzer = get_shap_analyzer()
        img_base64 = analyzer.generate_force_plot_base64(booking_data, model_name)

        if img_base64 is None:
            return jsonify({'error': '无法生成Force Plot图片'}), 500

        return jsonify({
            'image': img_base64,
            'format': 'png',
            'model': model_name
        })
    except Exception as e:
        return jsonify({'error': f'生成Force Plot失败: {str(e)}'}), 500

# ==================== 用户管理 API ====================

@app.route('/api/users', methods=['GET'])
@login_required
@role_required(['admin'])
def get_users():
    """获取所有用户列表（仅管理员）"""
    users = database.get_all_users()
    return jsonify({'users': users})

@app.route('/api/users/<int:user_id>', methods=['GET'])
@login_required
@role_required(['admin'])
def get_user(user_id):
    """获取用户详情（仅管理员）"""
    user = database.get_user_by_id(user_id)
    if user:
        # 不返回密码哈希
        user.pop('password_hash', None)
        return jsonify(user)
    return jsonify({'error': '用户不存在'}), 404

@app.route('/api/users', methods=['POST'])
@login_required
@role_required(['admin'])
def create_user():
    """创建新用户（仅管理员）"""
    data = request.json
    username = data.get('username')
    password = data.get('password')
    role = data.get('role', 'staff')

    # 验证必填字段
    if not username or not password:
        return jsonify({'error': '用户名和密码不能为空'}), 400

    # 验证角色合法性
    if role not in ['staff', 'manager', 'admin']:
        return jsonify({'error': '无效的角色类型'}), 400

    try:
        user_id = database.create_user(username, password, role)
        return jsonify({
            'id': user_id,
            'username': username,
            'role': role,
            'message': '用户创建成功'
        }), 201
    except Exception as e:
        if 'UNIQUE constraint failed' in str(e):
            return jsonify({'error': '用户名已存在'}), 409
        return jsonify({'error': str(e)}), 500

@app.route('/api/users/<int:user_id>', methods=['PUT'])
@login_required
@role_required(['admin'])
def update_user(user_id):
    """更新用户信息（仅管理员）"""
    data = request.json

    # 允许更新的字段
    allowed_fields = ['username', 'role', 'is_active']
    update_data = {k: v for k, v in data.items() if k in allowed_fields}

    # 如果有新密码，单独处理
    if 'password' in data and data['password']:
        from auth import hash_password
        update_data['password_hash'] = hash_password(data['password'])

    # 验证角色
    if 'role' in update_data and update_data['role'] not in ['staff', 'manager', 'admin']:
        return jsonify({'error': '无效的角色类型'}), 400

    try:
        success = database.update_user(user_id, update_data)
        if success:
            return jsonify({'message': '用户更新成功'})
        return jsonify({'error': '用户不存在'}), 404
    except Exception as e:
        if 'UNIQUE constraint failed' in str(e):
            return jsonify({'error': '用户名已存在'}), 409
        return jsonify({'error': str(e)}), 500

@app.route('/api/users/<int:user_id>', methods=['DELETE'])
@login_required
@role_required(['admin'])
def delete_user(user_id):
    """删除用户（仅管理员）"""
    try:
        success = database.delete_user(user_id)
        if success:
            return jsonify({'message': '用户删除成功'})
        return jsonify({'error': '用户不存在'}), 404
    except Exception as e:
        if '不能删除' in str(e) or 'cannot delete' in str(e).lower():
            return jsonify({'error': str(e)}), 400
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/password', methods=['PUT'])
@login_required
def change_password():
    """修改当前用户密码"""
    data = request.json
    old_password = data.get('old_password')
    new_password = data.get('new_password')

    if not old_password or not new_password:
        return jsonify({'error': '请提供原密码和新密码'}), 400

    user = request.current_user

    # 验证原密码
    if not database.verify_password(user['username'], old_password):
        return jsonify({'error': '原密码错误'}), 401

    # 更新密码
    from auth import hash_password
    success = database.update_user(user['id'], {'password_hash': hash_password(new_password)})

    if success:
        return jsonify({'message': '密码修改成功'})
    return jsonify({'error': '密码修改失败'}), 500

# ==================== 超参数优化API ====================

# 触发超参数优化（异步）
@app.route('/api/models/optimize', methods=['POST'])
@login_required
@role_required(['admin'])
def start_model_optimization():
    from hyperparameter_optimizer import start_optimization_async, get_optimization_status

    data = request.json
    model_name = data.get('model', 'XGBoost')
    n_trials = data.get('n_trials', 50)

    # 检查是否有正在运行的优化
    status = get_optimization_status()
    if status.get('is_running'):
        return jsonify({'error': '已有优化任务在运行中'}), 400

    start_optimization_async(model_name, n_trials)
    return jsonify({'message': f'已启动 {model_name} 超参数优化', 'trials': n_trials}), 202

# 查询优化进度
@app.route('/api/models/optimize/status', methods=['GET'])
@login_required
@role_required(['manager', 'admin'])
def get_optimize_status():
    from hyperparameter_optimizer import get_optimization_status
    status = get_optimization_status()
    return jsonify(status)

# 获取优化历史记录
@app.route('/api/models/optimize/history', methods=['GET'])
@login_required
@role_required(['manager', 'admin'])
def get_optimize_history():
    import os
    import json

    history = []
    models_dir = 'models'
    if os.path.exists(models_dir):
        for f in os.listdir(models_dir):
            if f.startswith('optimization_') and f.endswith('.json'):
                try:
                    with open(os.path.join(models_dir, f), 'r', encoding='utf-8') as fh:
                        log = json.load(fh)
                        history.append(log)
                except:
                    pass

    # 按时间倒序
    history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return jsonify({'history': history})

# ==================== 外部数据API ====================

# 获取天气数据
@app.route('/api/external/weather', methods=['GET'])
@login_required
@role_required(['staff', 'manager', 'admin'])
def get_weather_data():
    from external_data import WeatherDataService
    from datetime import datetime
    
    city = request.args.get('city', 'Lisbon')
    date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    
    service = WeatherDataService()
    weather = service.get_weather_by_city_date(city, date)
    return jsonify(weather)

# 检查是否为节假日
@app.route('/api/external/holiday', methods=['GET'])
@login_required
@role_required(['staff', 'manager', 'admin'])
def check_holiday():
    from external_data import HolidayCalendarService
    from datetime import datetime
    
    date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    region = request.args.get('region', 'ALL')
    
    service = HolidayCalendarService()
    is_hol, info = service.is_holiday(date, region)
    features = service.get_holiday_features(date)
    
    return jsonify({
        'is_holiday': is_hol,
        'holiday_info': info,
        'features': features
    })

# ==================== 收益优化API（超售策略） ====================

# 获取超售建议
@app.route('/api/revenue/overbooking', methods=['POST'])
@login_required
@role_required(['manager', 'admin'])
def get_overbooking_recommendation():
    from overbooking_engine import OverbookingEngine

    data = request.json
    bookings = data.get('bookings', [])       # 预订列表（含预测概率）
    rooms_available = data.get('rooms_available', 100)
    avg_adr = data.get('avg_adr')
    risk_pref = data.get('risk_preference', 'moderate')  # conservative/moderate/aggressive

    engine = OverbookingEngine()
    result = engine.calculate_overbooking_recommendation(
        bookings_with_predictions=bookings,
        total_rooms_available=rooms_available,
        avg_adr=avg_adr,
        risk_preference=risk_pref
    )

    return jsonify(result)

# 获取多场景对比
@app.route('/api/revenue/scenarios', methods=['POST'])
@login_required
@role_required(['manager', 'admin'])
def get_revenue_scenarios():
    from overbooking_engine import OverbookingEngine

    data = request.json
    bookings = data.get('bookings', [])
    rooms = data.get('rooms_available', 100)
    avg_adr = data.get('avg_adr')

    engine = OverbookingEngine()
    scenarios = engine.get_scenario_comparison(bookings, rooms, avg_adr)

    return jsonify(scenarios)

# 使用数据库中的当日预订自动计算超售建议
@app.route('/api/revenue/daily-analysis/<date_str>', methods=['GET'])
@login_required
@role_required(['manager', 'admin'])
def get_daily_revenue_analysis(date_str):
    """基于数据库中指定日期的预订数据进行分析"""
    import database
    from overbooking_engine import OverbookingEngine
    from prediction_service import get_prediction_service

    # 获取当日预订（这里简化处理，实际需要按日期筛选）
    bookings, total = database.get_all_bookings(limit=500)

    # 为每个预订获取预测概率
    pred_service = get_prediction_service()
    bookings_with_pred = []

    for b in bookings:
        try:
            # 将数据库记录转为预测输入格式
            booking_input = {k: v for k, v in b.items()
                           if k not in ['id', 'is_canceled', 'reservation_status', 'reservation_status_date']}

            # 获取预测概率
            pred_result = pred_service.predict(booking_input, 'XGBoost')

            bookings_with_pred.append({
                'booking_id': b.get('id'),
                'cancel_probability': pred_result.get('probability', {}).get('canceled', 0.37),
                'adr': b.get('adr', 100),
                **b
            })
        except Exception:
            bookings_with_pred.append({
                'booking_id': b.get('id'),
                'cancel_probability': 0.37,
                'adr': b.get('adr', 100),
                **b
            })

    engine = OverbookingEngine()
    result = engine.calculate_overbooking_recommendation(
        bookings_with_predictions=bookings_with_pred,
        total_rooms_available=request.args.get('rooms', 200, type=int),  # 默认200间房
        risk_preference=request.args.get('risk', 'moderate')
    )

    return jsonify(result)

# ==================== 高风险订单预警API ====================

# 获取预警列表 - 店员及以上可访问
@app.route('/api/alerts', methods=['GET'])
@login_required
@role_required(['staff', 'manager', 'admin'])
def get_alerts():
    service = AlertService()
    status_filter = request.args.get('status')  # pending/resolved/all
    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)

    alerts, total = service.get_alerts(status=status_filter, limit=limit, offset=offset)

    return jsonify({
        'alerts': alerts,
        'total': total,
        'limit': limit,
        'offset': offset
    })

# 获取预警统计 - 经理及以上可访问
@app.route('/api/alerts/statistics', methods=['GET'])
@login_required
@role_required(['manager', 'admin'])
def get_alert_statistics():
    service = AlertService()
    stats = service.get_alert_statistics()
    return jsonify(stats)

# 解决预警 - 店员及以上可访问
@app.route('/api/alerts/<int:alert_id>/resolve', methods=['PUT'])
@login_required
@role_required(['staff', 'manager', 'admin'])
def resolve_alert(alert_id):
    service = AlertService()
    data = request.json or {}
    note = data.get('note', '')

    success = service.resolve_alert(alert_id, note)
    if success:
        return jsonify({'message': '预警已解决'})
    return jsonify({'error': '预警不存在'}), 404

# 获取/更新预警配置 - 仅管理员可访问
@app.route('/api/alerts/config', methods=['GET', 'PUT'])
@login_required
@role_required(['admin'])
def alert_config():
    if request.method == 'GET':
        config = AlertService.get_alert_config()
        # 不返回敏感信息（密码等）
        safe_config = config.copy()
        if 'email_config' in safe_config:
            safe_config['email_config'] = {**safe_config['email_config'], 'sender_password': '******'}
        return jsonify(safe_config)

    else:  # PUT
        new_config = request.json
        AlertService.update_alert_config(new_config)
        return jsonify({'message': '预警配置已更新'})

# 手动触发预警检查（对现有预订批量扫描）- 仅管理员可访问
@app.route('/api/alerts/scan', methods=['POST'])
@login_required
@role_required(['admin'])
def trigger_alert_scan():
    import database as db_module
    from prediction_service import get_prediction_service
    from alert_service import AlertService

    limit = (request.json or {}).get('limit', 500)

    bookings, _ = db_module.get_all_bookings(limit=limit)
    pred_service = get_prediction_service()
    alert_service = AlertService()

    new_alerts = alert_service.batch_evaluate_bookings(bookings, pred_service)

    return jsonify({
        'scanned': len(bookings),
        'new_alerts_created': len(new_alerts),
        'alerts': new_alerts[:20]  # 返回前20条
    })

# ==================== MLflow 实验追踪API ====================

# 获取所有实验列表
@app.route('/api/experiments', methods=['GET'])
@login_required
@role_required(['manager', 'admin'])
def get_experiments():
    from mlflow_tracker import get_mlflow_tracker

    tracker = get_mlflow_tracker()
    model_filter = request.args.get('model')
    limit = request.args.get('limit', 50, type=int)

    experiments, total = tracker.get_all_experiments(model_name=model_filter, limit=limit)

    return jsonify({
        'experiments': experiments,
        'total': total
    })

# 获取实验详情
@app.route('/api/experiments/<experiment_id>', methods=['GET'])
@login_required
@role_required(['manager', 'admin'])
def get_experiment_detail(experiment_id):
    from mlflow_tracker import get_mlflow_tracker

    tracker = get_mlflow_tracker()
    experiment = tracker.get_experiment(experiment_id)

    if experiment:
        return jsonify(experiment)
    return jsonify({'error': '实验不存在'}), 404

# 获取最佳实验
@app.route('/api/experiments/best/<model_name>', methods=['GET'])
@login_required
@role_required(['manager', 'admin'])
def get_best_experiment(model_name):
    from mlflow_tracker import get_mlflow_tracker

    tracker = get_mlflow_tracker()
    best = tracker.get_best_experiment(model_name, metric=request.args.get('metric', 'roc_auc'))

    if best:
        return jsonify(best)
    return jsonify({'error': f'未找到 {model_name} 的实验记录'}), 404

# 对比实验
@app.route('/api/experiments/compare', methods=['POST'])
@login_required
@role_required(['manager', 'admin'])
def compare_experiments():
    from mlflow_tracker import get_mlflow_tracker

    data = request.json
    experiment_ids = data.get('experiment_ids', [])

    tracker = get_mlflow_tracker()
    comparisons = tracker.compare_experiments(experiment_ids)

    return jsonify(comparisons)

# 删除实验
@app.route('/api/experiments/<experiment_id>', methods=['DELETE'])
@login_required
@role_required(['admin'])
def delete_experiment(experiment_id):
    from mlflow_tracker import get_mlflow_tracker

    tracker = get_mlflow_tracker()
    success = tracker.delete_experiment(experiment_id)

    if success:
        return jsonify({'message': '实验已删除'})
    return jsonify({'error': '实验不存在'}), 404

# ==================== 缓存管理API ====================

# 获取缓存状态 - 仅管理员可访问
@app.route('/api/system/cache-stats', methods=['GET'])
@login_required
@role_required(['admin'])
def get_cache_stats_api():
    return jsonify(get_cache_stats())

# 清除缓存 - 仅管理员可访问
@app.route('/api/system/cache-clear', methods=['POST'])
@login_required
@role_required(['admin'])
def clear_cache():
    pattern = (request.json or {}).get('pattern')
    result = invalidate_cache(pattern)
    return jsonify(result)

# ==================== 系统状态 API ====================

# 获取数据库状态信息 - 仅管理员可访问
@app.route('/api/system/db-status', methods=['GET'])
@login_required
@role_required(['admin'])
def get_database_status():
    """获取数据库状态信息（数据库类型、表记录数等）"""
    try:
        from database_abstract import get_database_status
        status = get_database_status()
        return jsonify(status)
    except Exception:
        # 回退到简单检测（使用 SQLite）
        import sqlite3
        conn = sqlite3.connect('hotel_bookings.db')
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM bookings')
        total = cursor.fetchone()[0]
        conn.close()
        return jsonify({
            'db_type': 'sqlite',
            'db_info': {'type': 'sqlite', 'file': 'hotel_bookings.db'},
            'tables': {'bookings': total}
        })


# ==================== 概念漂移检测API（仅admin可访问）====================

# 获取漂移监控状态
@app.route('/api/monitoring/drift', methods=['GET'])
@login_required
@role_required(['admin'])
def get_drift_status():
    """获取概念漂移监控综合报告"""
    from drift_detector import get_drift_monitor

    monitor = get_drift_monitor()
    report = monitor.get_comprehensive_report()
    return jsonify(report)

# 设置基线
@app.route('/api/monitoring/baseline', methods=['POST'])
@login_required
@role_required(['admin'])
def set_baseline():
    """设置漂移检测基线分布"""
    from drift_detector import get_drift_monitor
    import pandas as pd
    import database

    monitor = get_drift_monitor()

    # 从数据库加载数据作为基线
    conn = sqlite3.connect(database.DB_FILE)
    df = pd.read_sql_query('SELECT * FROM bookings LIMIT 5000', conn)
    conn.close()

    snapshot_name = (request.json or {}).get('name', 'auto_baseline')
    baseline = monitor.set_baseline(df, snapshot_name=snapshot_name)

    return jsonify({
        'message': '基线已设置',
        'features_count': len(baseline),
        'snapshot_name': snapshot_name
    })

# 手动触发漂移检测
@app.route('/api/monitoring/drift/check', methods=['POST'])
@login_required
@role_required(['admin'])
def trigger_drift_check():
    """手动触发漂移检测"""
    from drift_detector import auto_check_and_alert
    result = auto_check_and_alert()
    return jsonify(result)

# 获取漂移事件历史
@app.route('/api/monitoring/drift/history', methods=['GET'])
@login_required
@role_required(['admin'])
def get_drift_history():
    """获取漂移事件历史记录"""
    from drift_detector import get_drift_monitor

    monitor = get_drift_monitor()
    limit = request.args.get('limit', 50, type=int)
    event_type = request.args.get('type')

    events = monitor.get_drift_history(limit=limit, event_type=event_type)
    return jsonify({'events': events})


# ==================== 启动配置 ====================

def open_browser():
    """在Flask启动后自动打开浏览器"""
    import time
    time.sleep(1.5)  # 等待服务器启动
    webbrowser.open('http://localhost:5001')

if __name__ == '__main__':
    print("启动酒店预订智能管理系统...")
    
    # 启动浏览器线程（仅在主进程执行，避免reloader重复打开）
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        threading.Thread(target=open_browser, daemon=True).start()
    
    app.run(debug=True, host='0.0.0.0', port=5001)

