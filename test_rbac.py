# -*- coding: utf-8 -*-
"""
RBAC 权限管理系统 - 全面集成测试脚本

测试范围：
1. 模块导入测试
2. 数据库初始化测试
3. 认证功能测试
4. 用户CRUD测试
5. 权限控制验证（各角色权限边界）
"""

import sys
import os
import io
import unittest
import json
import time
from datetime import datetime, timedelta

# 设置输出编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入被测模块
import database
import auth
from app import app


class TestResult:
    """测试结果收集器"""
    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0
        self.total = 0
    
    def add_result(self, test_name, passed, message=""):
        self.total += 1
        if passed:
            self.passed += 1
            status = "✓ 通过"
        else:
            self.failed += 1
            status = "✗ 失败"
        
        result = {
            'name': test_name,
            'status': status,
            'passed': passed,
            'message': message
        }
        self.results.append(result)
        
        # 实时输出结果
        print(f"  [{status}] {test_name}")
        if message and not passed:
            print(f"         原因: {message}")
    
    def print_summary(self):
        """打印测试总结"""
        print("\n" + "="*70)
        print("测试执行总结")
        print("="*70)
        print(f"总测试数: {self.total}")
        print(f"通过数量: {self.passed} ({self.passed/self.total*100:.1f}%)" if self.total > 0 else "通过数量: 0")
        print(f"失败数量: {self.failed} ({self.failed/self.total*100:.1f}%)" if self.total > 0 else "失败数量: 0")
        print("="*70)
        
        if self.failed > 0:
            print("\n失败的测试用例:")
            for r in self.results:
                if not r['passed']:
                    print(f"  - {r['name']}: {r['message']}")


class RBACIntegrationTest:
    """RBAC 集成测试类"""
    
    def __init__(self):
        self.result = TestResult()
        self.client = None
        self.admin_token = None
        self.staff_token = None
        self.manager_token = None
        self.test_user_ids = []  # 记录测试创建的用户ID，用于清理
    
    def setup(self):
        """测试环境初始化"""
        print("\n🔧 初始化测试环境...")
        
        # 初始化数据库
        try:
            database.init_db()
            print("  ✓ 数据库初始化成功")
        except Exception as e:
            print(f"  ✗ 数据库初始化失败: {e}")
            return False
        
        # 创建 Flask 测试客户端
        try:
            app.config['TESTING'] = True
            self.client = app.test_client()
            print("  ✓ Flask 测试客户端创建成功")
        except Exception as e:
            print(f"  ✗ Flask 测试客户端创建失败: {e}")
            return False
        
        return True
    
    def cleanup(self):
        """清理测试数据"""
        print("\n🧹 清理测试数据...")
        
        # 删除测试创建的用户
        for user_id in self.test_user_ids:
            try:
                if self.client and self.admin_token:
                    response = self.client.delete(
                        f'/api/users/{user_id}',
                        headers={'Authorization': f'Bearer {self.admin_token}'}
                    )
                    if response.status_code == 200:
                        print(f"  ✓ 已清理用户 ID: {user_id}")
            except Exception as e:
                print(f"  ✗ 清理用户 ID {user_id} 失败: {e}")
    
    # ==================== 1. 模块导入测试 ====================
    
    def test_module_imports(self):
        """测试模块是否可以正常导入"""
        print("\n📦 [1/5] 模块导入测试")
        print("-"*50)
        
        # 测试 auth 模块导入
        try:
            import auth
            self.result.add_result(
                "auth.py 模块导入",
                True,
                "auth 模块导入成功"
            )
            
            # 验证关键函数是否存在
            functions_to_check = [
                'generate_token', 'verify_token', 'login_required',
                'role_required', 'authenticate_user', 'hash_password',
                'verify_hash', 'get_current_user'
            ]
            
            for func_name in functions_to_check:
                has_func = hasattr(auth, func_name)
                self.result.add_result(
                    f"auth.{func_name} 函数存在",
                    has_func,
                    f"函数 {func_name} 不存在" if not has_func else ""
                )
                
        except ImportError as e:
            self.result.add_result(
                "auth.py 模块导入",
                False,
                f"导入失败: {e}"
            )
        
        # 测试 database 模块导入
        try:
            import database
            self.result.add_result(
                "database.py 模块导入",
                True,
                "database 模块导入成功"
            )
            
            # 验证用户相关函数是否存在
            user_functions = [
                'create_user', 'get_user_by_username', 'get_user_by_id',
                'get_all_users', 'update_user', 'delete_user', 'verify_password'
            ]
            
            for func_name in user_functions:
                has_func = hasattr(database, func_name)
                self.result.add_result(
                    f"database.{func_name} 函数存在",
                    has_func,
                    f"函数 {func_name} 不存在" if not has_func else ""
                )
                
        except ImportError as e:
            self.result.add_result(
                "database.py 模块导入",
                False,
                f"导入失败: {e}"
            )
    
    # ==================== 2. 数据库初始化测试 ====================
    
    def test_database_initialization(self):
        """测试数据库初始化和默认管理员账号"""
        print("\n💾 [2/5] 数据库初始化测试")
        print("-"*50)
        
        import sqlite3
        
        # 检查数据库文件是否存在
        db_exists = os.path.exists('hotel_bookings.db')
        self.result.add_result(
            "数据库文件存在",
            db_exists,
            "hotel_bookings.db 文件不存在" if not db_exists else ""
        )
        
        if db_exists:
            # 连接数据库并检查表结构
            try:
                conn = sqlite3.connect('hotel_bookings.db')
                cursor = conn.cursor()
                
                # 检查 users 表是否存在
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='users'
                """)
                table_exists = cursor.fetchone() is not None
                self.result.add_result(
                    "users 表存在",
                    table_exists,
                    "users 表不存在"
                )
                
                if table_exists:
                    # 检查表结构
                    cursor.execute("PRAGMA table_info(users)")
                    columns = [col[1] for col in cursor.fetchall()]
                    
                    required_columns = ['id', 'username', 'password_hash', 'role', 'is_active']
                    for col in required_columns:
                        col_exists = col in columns
                        self.result.add_result(
                            f"users 表包含 {col} 字段",
                            col_exists,
                            f"缺少字段: {col}" if not col_exists else ""
                        )
                    
                    # 检查默认管理员账号
                    cursor.execute("SELECT * FROM users WHERE username='admin'")
                    admin_user = cursor.fetchone()
                    
                    admin_exists = admin_user is not None
                    self.result.add_result(
                        "默认管理员账号 (admin) 存在",
                        admin_exists,
                        "默认管理员账号未创建"
                    )
                    
                    if admin_exists:
                        # 验证管理员角色
                        is_admin_role = admin_user[3] == 'admin'  # role 字段在第4列
                        self.result.add_result(
                            "管理员角色正确 (admin)",
                            is_admin_role,
                            f"角色不正确: {admin_user[3]}" if not is_admin_role else ""
                        )
                        
                        # 验证账号状态
                        is_active = admin_user[4] == 1  # is_active 字段在第5列
                        self.result.add_result(
                            "管理员账号已激活 (is_active=1)",
                            is_active,
                            "管理员账号未激活" if not is_active else ""
                        )
                        
                        # 验证密码可以正常验证
                        password_valid = database.verify_password('admin', 'admin123') is not None
                        self.result.add_result(
                            "管理员密码验证通过 (admin/admin123)",
                            password_valid,
                            "密码验证失败"
                        )
                
                conn.close()
                
            except sqlite3.Error as e:
                self.result.add_result(
                    "数据库连接和查询",
                    False,
                    f"数据库错误: {e}"
                )
    
    # ==================== 3. 认证功能测试 ====================
    
    def test_authentication(self):
        """测试认证功能"""
        print("\n🔐 [3/5] 认证功能测试")
        print("-"*50)
        
        # 测试正确密码登录
        response = self.client.post('/api/auth/login', json={
            'username': 'admin',
            'password': 'admin123'
        })
        
        login_success = response.status_code == 200
        self.result.add_result(
            "正确密码登录 - 返回200",
            login_success,
            f"返回状态码: {response.status_code}" if not login_success else ""
        )
        
        if login_success:
            data = response.get_json()
            has_token = 'token' in data
            self.result.add_result(
                "登录响应包含 token 字段",
                has_token,
                "响应中缺少 token 字段" if not has_token else ""
            )
            
            has_user_info = 'user' in data
            self.result.add_result(
                "登录响应包含 user 信息",
                has_user_info,
                "响应中缺少 user 字段" if not has_user_info else ""
            )
            
            if has_token:
                # 保存管理员token供后续使用
                self.admin_token = data['token']
                
                # 验证 Token 格式
                token_valid = len(data['token']) > 0
                self.result.add_result(
                    "Token 格式有效（非空字符串）",
                    token_valid,
                    "Token 为空或格式无效"
                )
                
                # 使用 verify_token 验证
                payload = auth.verify_token(data['token'])
                token_verified = payload is not None
                self.result.add_result(
                    "Token 可以通过 verify_token 验证",
                    token_verified,
                    "Token 验证失败" if not token_verified else ""
                )
                
                if token_verified:
                    # 验证 Token payload 内容
                    has_user_id = 'user_id' in payload
                    has_username = 'username' in payload
                    has_role = 'role' in payload
                    
                    self.result.add_result(
                        "Token 包含 user_id 字段",
                        has_user_id,
                        "缺少 user_id 字段"
                    )
                    self.result.add_result(
                        "Token 包含 username 字段",
                        has_username,
                        "缺少 username 字段"
                    )
                    self.result.add_result(
                        "Token 包含 role 字段",
                        has_role,
                        "缺少 role 字段"
                    )
                    
                    # 验证角色值
                    role_correct = payload.get('role') == 'admin' if has_role else False
                    self.result.add_result(
                        "Token 中角色值为 admin",
                        role_correct,
                        f"角色值错误: {payload.get('role')}" if has_role else "无role字段"
                    )
        
        # 测试错误密码登录
        response_wrong = self.client.post('/api/auth/login', json={
            'username': 'admin',
            'password': 'wrongpassword'
        })
        
        login_failed = response_wrong.status_code == 401
        self.result.add_result(
            "错误密码登录 - 返回401",
            login_failed,
            f"返回状态码: {response_wrong.status_code}, 期望401" if not login_failed else ""
        )
        
        # 测试不存在的用户登录
        response_notexist = self.client.post('/api/auth/login', json={
            'username': 'nonexistentuser',
            'password': 'anypassword'
        })
        
        not_exist_failed = response_notexist.status_code == 401
        self.result.add_result(
            "不存在用户登录 - 返回401",
            not_exist_failed,
            f"返回状态码: {response_notexist.status_code}, 期望401" if not not_exist_failed else ""
        )
        
        # 测试 Token 过期处理（生成一个立即过期的Token）
        if self.admin_token:
            # 创建一个已过期的Token进行测试
            from datetime import datetime, timedelta
            import jwt
            
            expired_payload = {
                'user_id': 1,
                'username': 'admin',
                'role': 'admin',
                'exp': datetime.utcnow() - timedelta(hours=1),  # 设置为1小时前过期
                'iat': datetime.utcnow() - timedelta(hours=25),
                'type': 'access'
            }
            
            expired_token = jwt.encode(
                expired_payload,
                auth.SECRET_KEY,
                algorithm=auth.JWT_ALGORITHM
            )
            
            # 尝试使用过期的Token访问受保护接口
            response_expired = self.client.get('/api/auth/me', headers={
                'Authorization': f'Bearer {expired_token}'
            })
            
            expired_rejected = response_expired.status_code == 401
            self.result.add_result(
                "过期Token访问 - 返回401",
                expired_rejected,
                f"返回状态码: {response_expired.status_code}, 期望401" if not expired_rejected else ""
            )
            
            # 验证 verify_token 对过期Token的处理
            verified_expired = auth.verify_token(expired_token) is None
            self.result.add_result(
                "verify_token 正确识别过期Token",
                verified_expired,
                "verify_token 未正确处理过期Token"
            )
        
        # 测试无效Token
        response_invalid = self.client.get('/api/auth/me', headers={
            'Authorization': 'Bearer invalid.token.here'
        })
        
        invalid_rejected = response_invalid.status_code == 401
        self.result.add_result(
            "无效Token访问 - 返回401",
            invalid_rejected,
            f"返回状态码: {response_invalid.status_code}, 期望401" if not invalid_rejected else ""
        )
    
    # ==================== 4. 用户 CRUD 测试 ====================
    
    def test_user_crud(self):
        """测试用户增删改查操作"""
        print("\n👥 [4/5] 用户 CRUD 测试")
        print("-"*50)
        
        if not self.admin_token:
            print("  ⚠ 跳过用户CRUD测试（无管理员Token）")
            return
        
        headers = {'Authorization': f'Bearer {self.admin_token}'}
        
        # 创建 staff 角色用户
        timestamp = int(time.time())
        staff_username = f'test_staff_{timestamp}'
        
        create_response = self.client.post('/api/users', json={
            'username': staff_username,
            'password': 'staff123',
            'role': 'staff'
        }, headers=headers)
        
        staff_created = create_response.status_code == 201
        self.result.add_result(
            "创建 staff 角色用户 - 返回201",
            staff_created,
            f"返回状态码: {create_response.status_code}" if not staff_created else ""
        )
        
        staff_user_id = None
        if staff_created:
            staff_data = create_response.get_json()
            staff_user_id = staff_data.get('id')
            self.test_user_ids.append(staff_user_id)
            
            # 验证返回的用户信息
            correct_username = staff_data.get('username') == staff_username
            correct_role = staff_data.get('role') == 'staff'
            
            self.result.add_result(
                "创建用户返回正确的用户名",
                correct_username,
                f"用户名不匹配: {staff_data.get('username')}"
            )
            self.result.add_result(
                "创建用户返回正确的角色",
                correct_role,
                f"角色不匹配: {staff_data.get('role')}"
            )
        
        # 创建 manager 角色用户
        manager_username = f'test_manager_{timestamp}'
        
        create_manager_response = self.client.post('/api/users', json={
            'username': manager_username,
            'password': 'manager123',
            'role': 'manager'
        }, headers=headers)
        
        manager_created = create_manager_response.status_code == 201
        self.result.add_result(
            "创建 manager 角色用户 - 返回201",
            manager_created,
            f"返回状态码: {create_manager_response.status_code}" if not manager_created else ""
        )
        
        manager_user_id = None
        if manager_created:
            manager_data = create_manager_response.get_json()
            manager_user_id = manager_data.get('id')
            self.test_user_ids.append(manager_user_id)
            
            correct_manager_role = manager_data.get('role') == 'manager'
            self.result.add_result(
                "Manager 用户角色正确",
                correct_manager_role,
                f"角色不匹配: {manager_data.get('role')}"
            )
        
        # 查询用户列表
        list_response = self.client.get('/api/users', headers=headers)
        list_success = list_response.status_code == 200
        self.result.add_result(
            "查询用户列表 - 返回200",
            list_success,
            f"返回状态码: {list_response.status_code}" if not list_success else ""
        )
        
        if list_success:
            users_data = list_response.get_json()
            has_users_list = 'users' in users_data
            self.result.add_result(
                "用户列表响应包含 users 字段",
                has_users_list,
                "缺少 users 字段"
            )
            
            if has_users_list:
                users_list = users_data['users']
                is_list_type = isinstance(users_list, list)
                self.result.add_result(
                    "users 字段是列表类型",
                    is_list_type,
                    f"类型错误: {type(users_list)}"
                )
                
                # 验证新创建的用户在列表中
                if staff_user_id and is_list_type:
                    created_staff_in_list = any(
                        u.get('id') == staff_user_id for u in users_list
                    )
                    self.result.add_result(
                        "新创建的 staff 用户在列表中",
                        created_staff_in_list,
                        "未在用户列表中找到新建的staff用户"
                    )
        
        # 查询用户详情
        if staff_user_id:
            detail_response = self.client.get(f'/api/users/{staff_user_id}', headers=headers)
            detail_success = detail_response.status_code == 200
            self.result.add_result(
                "查询用户详情 - 返回200",
                detail_success,
                f"返回状态码: {detail_response.status_code}" if not detail_success else ""
            )
            
            if detail_success:
                user_detail = detail_response.get_json()
                
                # 确保不返回密码哈希
                no_password = 'password_hash' not in user_detail
                self.result.add_result(
                    "用户详情不包含密码哈希",
                    no_password,
                    "安全风险: 响应中包含密码哈希"
                )
                
                # 验证用户信息正确
                correct_id = user_detail.get('id') == staff_user_id
                correct_detail_username = user_detail.get('username') == staff_username
                
                self.result.add_result(
                    "用户详情ID正确",
                    correct_id,
                    f"ID不匹配: {user_detail.get('id')}"
                )
                self.result.add_result(
                    "用户详情用户名正确",
                    correct_detail_username,
                    f"用户名不匹配: {user_detail.get('username')}"
                )
        
        # 更新用户信息
        if staff_user_id:
            update_username = f'{staff_username}_updated'
            update_response = self.client.put(f'/api/users/{staff_user_id}', json={
                'username': update_username,
                'is_active': True
            }, headers=headers)
            
            update_success = update_response.status_code == 200
            self.result.add_result(
                "更新用户信息 - 返回200",
                update_success,
                f"返回状态码: {update_response.status_code}" if not update_success else ""
            )
            
            if update_success:
                # 验证更新是否生效
                verify_response = self.client.get(f'/api/users/{staff_user_id}', headers=headers)
                if verify_response.status_code == 200:
                    updated_user = verify_response.get_json()
                    username_updated = updated_user.get('username') == update_username
                    self.result.add_result(
                        "用户名更新生效",
                        username_updated,
                        f"更新后用户名: {updated_user.get('username')}"
                    )
        
        # 创建重复用户名（应失败）
        duplicate_response = self.client.post('/api/users', json={
            'username': 'admin',  # 已存在的用户名
            'password': 'newpass123',
            'role': 'staff'
        }, headers=headers)
        
        # 注意：database.create_user 在用户名存在时返回None，
        # 但 app.py 的 create_user 路由可能返回不同的状态码
        duplicate_failed = duplicate_response.status_code in [409, 500, 400]
        self.result.add_result(
            "创建重复用户名 - 返回错误状态码",
            duplicate_failed,
            f"返回状态码: {duplicate_response.status_code}, 期望409/500/400" if not duplicate_failed else ""
        )
        
        # 删除用户
        if manager_user_id:
            delete_response = self.client.delete(f'/api/users/{manager_user_id}', headers=headers)
            delete_success = delete_response.status_code == 200
            self.result.add_result(
                "删除用户 - 返回200",
                delete_success,
                f"返回状态码: {delete_response.status_code}" if not delete_success else ""
            )
            
            if delete_success:
                # 验证删除后无法查询
                verify_delete = self.client.get(f'/api/users/{manager_user_id}', headers=headers)
                deleted_not_found = verify_delete.status_code == 404
                self.result.add_result(
                    "删除后查询用户 - 返回404",
                    deleted_not_found,
                    f"返回状态码: {verify_delete.status_code}, 期望404" if not deleted_not_found else ""
                )
                
                # 从清理列表中移除已删除的用户
                if manager_user_id in self.test_user_ids:
                    self.test_user_ids.remove(manager_user_id)
    
    # ==================== 5. 权限控制验证 ====================
    
    def test_permission_control(self):
        """测试各角色的权限边界"""
        print("\n🔒 [5/5] 权限控制验证")
        print("-"*50)
        
        if not self.admin_token:
            print("  ⚠ 跳过权限控制测试（无管理员Token）")
            return
        
        # 获取各角色的Token
        # 先获取staff用户的token
        staff_login = self.client.post('/api/auth/login', json={
            'username': 'test_staff_' + str(int(time.time()) - 10).split('.')[-1][:10],  # 可能需要调整
            'password': 'staff123'
        })
        
        # 如果上面的方法不行，直接查找测试创建的staff用户
        # 通过API获取用户列表找到staff用户
        users_response = self.client.get('/api/users', headers={'Authorization': f'Bearer {self.admin_token}'})
        if users_response.status_code == 200:
            users = users_response.get_json().get('users', [])
            staff_user = next((u for u in users if u.get('role') == 'staff' and u.get('username').startswith('test_')), None)
            
            if staff_user:
                staff_login_resp = self.client.post('/api/auth/login', json={
                    'username': staff_user['username'],
                    'password': 'staff123'
                })
                if staff_login_resp.status_code == 200:
                    self.staff_token = staff_login_resp.get_json().get('token')
        
        # 创建一个新的manager用户并获取token
        timestamp = int(time.time())
        manager_test_user = f'mgr_test_{timestamp}'
        create_mgr = self.client.post('/api/users', json={
            'username': manager_test_user,
            'password': 'mgr123',
            'role': 'manager'
        }, headers={'Authorization': f'Bearer {self.admin_token}'})
        
        if create_mgr.status_code == 201:
            mgr_id = create_mgr.get_json().get('id')
            self.test_user_ids.append(mgr_id)
            
            mgr_login = self.client.post('/api/auth/login', json={
                'username': manager_test_user,
                'password': 'mgr123'
            })
            if mgr_login.status_code == 200:
                self.manager_token = mgr_login.get_json().get('token')
        
        # ===== 店员(staff)权限测试 =====
        print("\n  📋 店员(Staff)权限边界测试:")
        
        if self.staff_token:
            staff_headers = {'Authorization': f'Bearer {self.staff_token}'}
            
            # 应该允许的接口
            allowed_endpoints = [
                ('GET', '/api/bookings'),
                ('GET', '/api/statistics'),
                ('GET', '/api/auth/me'),
            ]
            
            for method, endpoint in allowed_endpoints:
                if method == 'GET':
                    resp = self.client.get(endpoint, headers=staff_headers)
                elif method == 'POST':
                    resp = self.client.post(endpoint, headers=staff_headers)
                
                allowed = resp.status_code in [200, 404]  # 404表示路由存在但数据不存在
                self.result.add_result(
                    f"[Staff] 可访问 {method} {endpoint}",
                    allowed,
                    f"返回状态码: {resp.status_code}, 期望200或404" if not allowed else ""
                )
            
            # 应该禁止的接口
            forbidden_endpoints = [
                ('POST', '/api/predict', {}),
                ('GET', '/api/users', None),
                ('POST', '/api/users', {'username': 'x', 'password': 'y'}),
                ('GET', '/api/models', None),
            ]
            
            for method, endpoint, data in forbidden_endpoints:
                if method == 'GET':
                    resp = self.client.get(endpoint, headers=staff_headers)
                elif method == 'POST':
                    resp = self.client.post(endpoint, json=data or {}, headers=staff_headers)
                
                forbidden = resp.status_code == 403
                self.result.add_result(
                    f"[Staff] 禁止访问 {method} {endpoint}",
                    forbidden,
                    f"返回状态码: {resp.status_code}, 期望403" if not forbidden else ""
                )
        else:
            print("    ⚠ 无Staff Token，跳过Staff权限测试")
        
        # ===== 经理(manager)权限测试 =====
        print("\n  👔 经理(Manager)权限边界测试:")
        
        if self.manager_token:
            mgr_headers = {'Authorization': f'Bearer {self.manager_token}'}
            
            # 应该允许的接口
            mgr_allowed = [
                ('GET', '/api/bookings', None),
                ('GET', '/api/statistics', None),
                ('POST', '/api/predict', {'booking': {}}),
                ('GET', '/api/models', None),
                ('GET', '/api/models/performance', None),
                ('GET', '/api/auth/me', None),
            ]
            
            for method, endpoint, data in mgr_allowed:
                if method == 'GET':
                    resp = self.client.get(endpoint, headers=mgr_headers)
                elif method == 'POST':
                    resp = self.client.post(endpoint, json=data or {}, headers=mgr_headers)
                
                allowed = resp.status_code in [200, 400, 500, 404]  # 允许业务逻辑错误，但不应该是权限错误
                self.result.add_result(
                    f"[Manager] 可访问 {method} {endpoint}",
                    allowed,
                    f"返回状态码: {resp.status_code}, 不应为403" if not allowed else ""
                )
            
            # 应该禁止的接口
            mgr_forbidden = [
                ('GET', '/api/users', None),
                ('POST', '/api/users', {'username': 'x', 'password': 'y'}),
                ('DELETE', '/api/users/999', None),
            ]
            
            for method, endpoint, data in mgr_forbidden:
                if method == 'GET':
                    resp = self.client.get(endpoint, headers=mgr_headers)
                elif method == 'POST':
                    resp = self.client.post(endpoint, json=data or {}, headers=mgr_headers)
                elif method == 'DELETE':
                    resp = self.client.delete(endpoint, headers=mgr_headers)
                
                forbidden = resp.status_code == 403
                self.result.add_result(
                    f"[Manager] 禁止访问 {method} {endpoint}",
                    forbidden,
                    f"返回状态码: {resp.status_code}, 期望403" if not forbidden else ""
                )
        else:
            print("    ⚠ 无Manager Token，跳过Manager权限测试")
        
        # ===== 管理员(admin)权限测试 =====
        print("\n  🛡️ 管理员(Admin)权限边界测试:")
        
        admin_headers = {'Authorization': f'Bearer {self.admin_token}'}
        
        # 管理员应该能够访问所有接口
        admin_should_access = [
            ('GET', '/api/bookings', None),
            ('GET', '/api/statistics', None),
            ('POST', '/api/predict', {'booking': {}}),
            ('GET', '/api/models', None),
            ('GET', '/api/users', None),
            ('GET', '/api/auth/me', None),
        ]
        
        for method, endpoint, data in admin_should_access:
            if method == 'GET':
                resp = self.client.get(endpoint, headers=admin_headers)
            elif method == 'POST':
                resp = self.client.post(endpoint, json=data or {}, headers=admin_headers)
            
            accessible = resp.status_code in [200, 400, 500, 201, 404]
            self.result.add_result(
                f"[Admin] 可访问 {method} {endpoint}",
                accessible,
                f"返回状态码: {resp.status_code}, Admin应有完全访问权" if not accessible else ""
            )
        
        # ===== 未认证访问测试 =====
        print("\n  🚫 未认证访问测试:")
        
        unauthenticated_endpoints = [
            ('GET', '/api/bookings'),
            ('GET', '/api/statistics'),
            ('POST', '/api/predict'),
            ('GET', '/api/users'),
            ('GET', '/api/models'),
            ('GET', '/api/auth/me'),
        ]
        
        for method, endpoint in unauthenticated_endpoints:
            if method == 'GET':
                resp = self.client.get(endpoint)
            elif method == 'POST':
                resp = self.client.post(endpoint)
            
            unauthorized = resp.status_code in [401, 403]
            self.result.add_result(
                f"[未认证] 访问 {method} {endpoint} - 返回401/403",
                unauthorized,
                f"返回状态码: {resp.status_code}, 期望401或403" if not unauthorized else ""
            )


def main():
    """主测试入口"""
    print("="*70)
    print("🧪 RBAC 权限管理系统 - 全面集成测试")
    print("="*70)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"项目目录: {os.getcwd()}")
    
    # 创建测试实例
    tester = RBACIntegrationTest()
    
    # 初始化测试环境
    if not tester.setup():
        print("\n❌ 测试环境初始化失败，终止测试")
        return
    
    try:
        # 执行所有测试
        tester.test_module_imports()
        tester.test_database_initialization()
        tester.test_authentication()
        tester.test_user_crud()
        tester.test_permission_control()
        
        # 输出测试总结
        tester.result.print_summary()
        
    finally:
        # 清理测试数据
        tester.cleanup()
    
    # 返回退出码（0表示全部通过，1表示有失败）
    return 0 if tester.result.failed == 0 else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
