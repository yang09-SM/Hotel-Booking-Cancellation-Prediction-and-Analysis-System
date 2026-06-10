"""
认证与授权模块 (RBAC权限管理)
提供用户登录验证、JWT Token管理、角色权限控制等功能
"""

import os
from functools import wraps
from datetime import datetime, timedelta
from flask import request, jsonify, current_app
import jwt
from werkzeug.security import generate_password_hash, check_password_hash

# ==================== 配置常量 ====================

# JWT密钥（优先从环境变量获取，否则使用默认值）
SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'hotel_booking_secret_key_2024')

# Token过期时间：24小时
TOKEN_EXPIRE_HOURS = 24

# Token算法
JWT_ALGORITHM = 'HS256'


# ==================== 密码哈希工具函数 ====================

def hash_password(password):
    """
    对密码进行哈希处理
    参数:
        password: 明文密码
    返回:
        哈希后的密码字符串
    """
    return generate_password_hash(password)


def verify_hash(password, password_hash):
    """
    验证密码是否匹配哈希值
    参数:
        password: 明文密码
        password_hash: 哈希后的密码
    返回:
        是否匹配（布尔值）
    """
    return check_password_hash(password_hash, password)


# ==================== JWT Token管理 ====================

def generate_token(user_id, username, role):
    """
    生成JWT Token
    参数:
        user_id: 用户ID
        username: 用户名
        role: 用户角色
    返回:
        JWT Token字符串
    """
    # 设置Token过期时间
    expire_time = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    
    # 构建Token载荷（Payload）
    payload = {
        'user_id': user_id,
        'username': username,
        'role': role,
        'exp': expire_time,
        'iat': datetime.utcnow(),  # 签发时间
        'type': 'access'  # Token类型
    }
    
    # 生成Token
    token = jwt.encode(
        payload,
        SECRET_KEY,
        algorithm=JWT_ALGORITHM
    )
    
    return token


def verify_token(token):
    """
    验证并解码JWT Token
    参数:
        token: JWT Token字符串
    返回:
        成功返回解码后的payload字典，失败返回None
    """
    try:
        # 解码并验证Token
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[JWT_ALGORITHM]
        )
        
        return payload
        
    except jwt.ExpiredSignatureError:
        print("Token已过期")
        return None
    except jwt.InvalidTokenError as e:
        print(f"无效的Token: {e}")
        return None


def get_token_from_request():
    """
    从请求头中提取Token
    返回:
        Token字符串，如果不存在则返回None
    """
    auth_header = request.headers.get('Authorization')
    
    if not auth_header:
        return None
    
    # 支持格式: "Bearer <token>"
    parts = auth_header.split()
    
    if len(parts) != 2 or parts[0].lower() != 'bearer':
        return None
    
    return parts[1]


def get_current_user():
    """
    获取当前登录用户信息
    返回:
        用户信息字典，如果未登录则返回None
    """
    token = get_token_from_request()
    
    if not token:
        return None
    
    payload = verify_token(token)
    
    if not payload:
        return None
    
    return {
        'user_id': payload.get('user_id'),
        'username': payload.get('username'),
        'role': payload.get('role')
    }


# ==================== 装饰器 ====================

def login_required(f):
    """
    登录验证装饰器
    用于保护需要登录才能访问的路由
    用法:
        @app.route('/api/protected')
        @login_required
        def protected_route():
            ...
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # 获取当前用户
        current_user = get_current_user()
        
        if not current_user:
            return jsonify({
                'error': '未授权访问',
                'message': '请先登录',
                'code': 'AUTH_REQUIRED'
            }), 401
        
        # 将用户信息注入到请求上下文中
        request.current_user = current_user
        
        return f(*args, **kwargs)
    
    return decorated_function


def role_required(roles):
    """
    角色权限装饰器
    用于限制特定角色的用户才能访问路由
    参数:
        roles: 允许的角色列表，例如 ['admin', 'manager']
    用法:
        @app.route('/api/admin-only')
        @login_required
        @role_required(['admin'])
        def admin_route():
            ...
        
        @app.route('/api/manager-or-above')
        @login_required
        @role_required(['manager', 'admin'])
        def manager_route():
            ...
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # 确保用户已登录
            current_user = get_current_user()
            
            if not current_user:
                return jsonify({
                    'error': '未授权访问',
                    'message': '请先登录',
                    'code': 'AUTH_REQUIRED'
                }), 401
            
            # 检查用户角色是否有权限
            user_role = current_user.get('role')
            
            if user_role not in roles:
                return jsonify({
                    'error': '权限不足',
                    'message': f'需要以下角色之一: {", ".join(roles)}，当前角色: {user_role}',
                    'code': 'FORBIDDEN'
                }), 403
            
            # 将用户信息注入到请求上下文
            request.current_user = current_user
            
            return f(*args, **kwargs)
        
        return decorated_function
    
    return decorator


# ==================== 辅助函数 ====================

def is_admin():
    """检查当前用户是否是管理员"""
    current_user = get_current_user()
    return current_user and current_user.get('role') == 'admin'


def is_manager_or_above():
    """检查当前用户是否是经理或管理员"""
    current_user = get_current_user()
    return current_user and current_user.get('role') in ['manager', 'admin']


def require_permission(required_role):
    """
    权限检查辅助函数（非装饰器版本）
    可在视图函数内部手动调用进行权限检查
    参数:
        required_role: 所需的最低角色等级 ('staff', 'manager', 'admin')
    返回:
        是否有权限（布尔值）
    """
    # 角色权限等级映射
    role_hierarchy = {
        'staff': 1,
        'manager': 2,
        'admin': 3
    }
    
    current_user = get_current_user()
    
    if not current_user:
        return False
    
    user_role_level = role_hierarchy.get(current_user.get('role'), 0)
    required_role_level = role_hierarchy.get(required_role, 0)
    
    return user_role_level >= required_role_level


# ==================== 认证API辅助函数 ====================

def authenticate_user(username, password):
    """
    用户认证函数
    验证用户名和密码，成功则返回Token
    参数:
        username: 用户名
        password: 明文密码
    返回:
        成功返回 (True, token, user_info)
        失败返回 (False, error_message, None)
    """
    import database
    
    # 验证用户凭证
    user = database.verify_password(username, password)
    
    if not user:
        return False, '用户名或密码错误', None
    
    # 检查账号是否激活
    if not user.get('is_active'):
        return False, '账号已被禁用，请联系管理员', None
    
    # 生成Token
    token = generate_token(
        user_id=user['id'],
        username=user['username'],
        role=user['role']
    )
    
    # 准备返回的用户信息（不包含敏感数据）
    user_info = {
        'id': user['id'],
        'username': user['username'],
        'role': user['role'],
        'token': token,
        'expires_in': TOKEN_EXPIRE_HOURS * 3600  # 过期时间（秒）
    }
    
    return True, '登录成功', user_info


def logout_user():
    """
    用户登出（客户端处理，此函数用于记录日志等扩展）
    返回:
        登出消息字典
    """
    # 在实际应用中，可以将Token加入黑名单
    # 这里仅返回成功消息，由客户端删除Token
    
    return {
        'message': '登出成功',
        'code': 'LOGOUT_SUCCESS'
    }
