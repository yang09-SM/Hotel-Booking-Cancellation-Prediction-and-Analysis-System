"""
数据库抽象层 - 支持 SQLite / PostgreSQL 双引擎切换
通过环境变量 DATABASE_URL 控制使用哪种数据库
SQLite 用于开发/测试，PostgreSQL 用于生产环境
"""

import os
import sqlite3
import pandas as pd
from datetime import datetime

# 数据库配置
DATABASE_URL = os.environ.get('DATABASE_URL', '')  # 如 postgresql://user:pass@localhost:5432/hotel_db
USE_SQLITE = not DATABASE_URL or DATABASE_URL.startswith('sqlite')
DB_FILE = os.environ.get('DB_FILE', 'hotel_bookings.db')


class DatabaseManager:
    """数据库管理器 - 统一接口"""

    def __init__(self):
        self.db_type = 'postgresql' if DATABASE_URL and not USE_SQLITE else 'sqlite'
        self._connection_params = self._get_connection_params()

    def _get_connection_params(self):
        """获取连接参数"""
        if self.db_type == 'postgresql':
            # 解析 DATABASE_URL
            # 格式: postgresql://user:password@host:port/database
            url = DATABASE_URL
            if '://' in url:
                rest = url.split('://')[1]
                if '@' in rest:
                    cred, host_part = rest.split('@', 1)
                    if ':' in cred:
                        user, password = cred.split(':', 1)
                    else:
                        user, password = cred, ''

                    if '/' in host_part:
                        host_port, dbname = host_part.rsplit('/', 1)
                        if ':' in host_port:
                            host, port = host_port.rsplit(':', 1)
                            port = int(port)
                        else:
                            host, port = host_port, 5432
                    else:
                        host, port, dbname = 'localhost', 5432, 'hotel_db'

                    return {
                        'host': host, 'port': port, 'dbname': dbname,
                        'user': user, 'password': password
                    }
            return {'host': 'localhost', 'port': 5432, 'dbname': 'hotel_db',
                    'user': 'postgres', 'password': ''}
        else:
            return {'db_file': DB_FILE}

    def get_connection(self):
        """获取数据库连接"""
        if self.db_type == 'postgresql':
            try:
                import psycopg2
                from psycopg2.extras import RealDictCursor
                conn = psycopg2.connect(**self._connection_params)
                return conn
            except ImportError:
                print("警告: psycopg2 未安装，回退到 SQLite")
                self.db_type = 'sqlite'
                return self._get_sqlite_connection()
        else:
            return self._get_sqlite_connection()

    def _get_sqlite_connection(self):
        """获取 SQLite 连接"""
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        return conn

    @property
    def db_info(self):
        """返回当前数据库信息"""
        return {
            'type': self.db_type,
            'file' if self.db_type == 'sqlite' else 'url': (
                DB_FILE if self.db_type == 'sqlite' else DATABASE_URL
            )
        }

    def get_placeholder(self):
        """获取当前数据库的参数占位符"""
        return '%s' if self.db_type == 'postgresql' else '?'

    def execute_fetchall(self, cursor, query, params=None):
        """执行查询并返回所有结果（统一处理行格式）"""
        if params is None:
            cursor.execute(query)
        else:
            cursor.execute(query, params)
        rows = cursor.fetchall()
        # 统一转换为字典列表
        if self.db_type == 'postgresql':
            # psycopg2 的 RealDictCursor 返回字典，普通游标返回元组
            if rows and isinstance(rows[0], dict):
                return [dict(row) for row in rows]
            else:
                desc = cursor.description
                if desc:
                    col_names = [col[0] for col in desc]
                    return [dict(zip(col_names, row)) for row in rows]
                return [dict(row) for row in rows]
        else:
            # SQLite Row 对象转字典
            return [dict(row) for row in rows]

    def execute_fetchone(self, cursor, query, params=None):
        """执行查询并返回单条结果"""
        if params is None:
            cursor.execute(query)
        else:
            cursor.execute(query, params)
        row = cursor.fetchone()
        if row is None:
            return None
        if self.db_type == 'postgresql':
            if isinstance(row, dict):
                return dict(row)
            desc = cursor.description
            if desc:
                col_names = [col[0] for col in desc]
                return dict(zip(col_names, row))
            return dict(row)
        else:
            return dict(row)

    def get_lastrowid(self, cursor):
        """获取最后插入行的 ID（兼容两种数据库）"""
        if self.db_type == 'postgresql':
            cursor.execute("SELECT lastval()")
            return cursor.fetchone()[0]
        else:
            return cursor.lastrowid


# ===== 全局数据库管理器实例 =====
_db_manager = DatabaseManager()


# ==================== 数据库初始化函数 ====================

def init_db():
    """初始化数据库表结构"""
    from werkzeug.security import generate_password_hash

    conn = _db_manager.get_connection()
    cursor = conn.cursor()

    ph = _db_manager.get_placeholder()

    # 预订表 DDL（兼容 SQLite 和 PostgreSQL）
    # 注意：PostgreSQL 使用 SERIAL，SQLite 使用 INTEGER PRIMARY KEY AUTOINCREMENT
    # 这里使用兼容写法
    if _db_manager.db_type == 'postgresql':
        bookings_ddl = '''
            CREATE TABLE IF NOT EXISTS bookings (
                id SERIAL PRIMARY KEY,
                hotel TEXT,
                is_canceled INTEGER,
                lead_time INTEGER,
                arrival_date_year INTEGER,
                arrival_date_month TEXT,
                arrival_date_week_number INTEGER,
                arrival_date_day_of_month INTEGER,
                stays_in_weekend_nights INTEGER,
                stays_in_week_nights INTEGER,
                adults INTEGER,
                children REAL,
                babies INTEGER,
                meal TEXT,
                country TEXT,
                market_segment TEXT,
                distribution_channel TEXT,
                is_repeated_guest INTEGER,
                previous_cancellations INTEGER,
                previous_bookings_not_canceled INTEGER,
                reserved_room_type TEXT,
                assigned_room_type TEXT,
                booking_changes INTEGER,
                deposit_type TEXT,
                agent REAL,
                company REAL,
                days_in_waiting_list INTEGER,
                customer_type TEXT,
                adr REAL,
                required_car_parking_spaces INTEGER,
                total_of_special_requests INTEGER,
                reservation_status TEXT,
                reservation_status_date TEXT
            )
        '''
    else:
        bookings_ddl = '''
            CREATE TABLE IF NOT EXISTS bookings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hotel TEXT,
                is_canceled INTEGER,
                lead_time INTEGER,
                arrival_date_year INTEGER,
                arrival_date_month INTEGER,
                stays_in_weekend_nights INTEGER,
                stays_in_week_nights INTEGER,
                adults INTEGER,
                children INTEGER,
                babies INTEGER,
                meal TEXT,
                country TEXT,
                market_segment TEXT,
                distribution_channel TEXT,
                is_repeated_guest INTEGER,
                previous_cancellations INTEGER,
                previous_bookings_not_canceled INTEGER,
                reserved_room_type TEXT,
                assigned_room_type TEXT,
                booking_changes INTEGER,
                deposit_type TEXT,
                agent INTEGER,
                company INTEGER,
                days_in_waiting_list INTEGER,
                customer_type TEXT,
                adr REAL,
                required_car_parking_spaces INTEGER,
                total_of_special_requests INTEGER,
                reservation_status TEXT,
                reservation_status_date TEXT,
                arrival_date_week_number INTEGER,
                arrival_date_day_of_month INTEGER
            )
        '''

    cursor.execute(bookings_ddl)

    # 用户表 DDL
    if _db_manager.db_type == 'postgresql':
        users_ddl = '''
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(100) UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role VARCHAR(20) NOT NULL CHECK(role IN ('staff', 'manager', 'admin')),
                is_active INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        '''
    else:
        users_ddl = '''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('staff', 'manager', 'admin')),
                is_active INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        '''
    cursor.execute(users_ddl)

    conn.commit()
    conn.close()

    # 初始化默认用户
    init_default_admin()


def init_default_admin():
    """初始化默认用户账号（仅在users表为空时）"""
    from werkzeug.security import generate_password_hash

    conn = _db_manager.get_connection()
    cursor = conn.cursor()

    ph = _db_manager.get_placeholder()

    # 检查是否已有用户
    cursor.execute('SELECT COUNT(*) FROM users')
    user_count = cursor.fetchone()[0]

    if user_count == 0:
        # 创建默认用户账号
        default_users = [
            ('admin', generate_password_hash('admin123'), 'admin'),
            ('manager', generate_password_hash('manager123'), 'manager'),
            ('staff', generate_password_hash('staff123'), 'staff'),
        ]
        placeholders = ', '.join([ph] * 3)
        cursor.executemany(f'''
            INSERT INTO users (username, password_hash, role)
            VALUES ({placeholders})
        ''', default_users)

        print("已创建默认账号:")
        print("  管理员: admin / admin123")
        print("  经理:   manager / manager123")
        print("  店员:   staff / staff123")

        conn.commit()

    conn.close()


# ==================== 预订管理相关函数 ====================

def import_csv_to_db(csv_path):
    """导入 CSV 到数据库"""
    if not os.path.exists(DB_FILE) and _db_manager.db_type == 'sqlite':
        init_db()

    df = pd.read_csv(csv_path)
    conn = _db_manager.get_connection()

    if _db_manager.db_type == 'postgresql':
        # PostgreSQL 使用 COPY 或逐行 INSERT
        from io import StringIO
        buffer = StringIO()
        df.to_csv(buffer, index=False, header=False)
        buffer.seek(0)
        cursor = conn.cursor()
        cursor.copy_from(buffer, 'bookings', sep=',', null='')
        conn.commit()
    else:
        df.to_sql('bookings', conn, if_exists='replace', index=False)

    conn.close()
    print(f"成功导入 {len(df)} 条记录到数据库")


def get_all_bookings(limit=100, offset=0):
    """获取所有预订记录（分页）"""
    conn = _db_manager.get_connection()
    cursor = conn.cursor()

    ph = _db_manager.get_placeholder()

    cursor.execute(f'SELECT * FROM bookings ORDER BY id DESC LIMIT {ph} OFFSET {ph}', (limit, offset))
    rows = cursor.fetchall()

    cursor.execute('SELECT COUNT(*) FROM bookings')
    total = cursor.fetchone()[0]

    conn.close()

    # 统一转换为字典列表
    bookings = _db_manager.execute_fetchall(cursor.__class__, 'SELECT * FROM bookings LIMIT 1') if False else []
    if _db_manager.db_type == 'postgresql':
        if rows:
            desc = cursor.description
            col_names = [col[0] for col in desc] if desc else []
            bookings = [dict(zip(col_names, row)) for row in rows]
        else:
            bookings = []
    else:
        bookings = [dict(row) for row in rows]

    return bookings, total


def get_booking_by_id(booking_id):
    """根据ID获取预订详情"""
    conn = _db_manager.get_connection()
    cursor = conn.cursor()

    ph = _db_manager.get_placeholder()

    cursor.execute(f'SELECT * FROM bookings WHERE id = {ph}', (booking_id,))
    row = cursor.fetchone()

    conn.close()

    if row is None:
        return None

    if _db_manager.db_type == 'postgresql':
        desc = cursor.description
        col_names = [col[0] for col in desc] if desc else []
        return dict(zip(col_names, row))
    else:
        return dict(row)


def create_booking(booking_data):
    """创建新预订"""
    conn = _db_manager.get_connection()
    cursor = conn.cursor()

    ph = _db_manager.get_placeholder()

    columns = ', '.join(booking_data.keys())
    placeholders = ', '.join([ph] * len(booking_data))
    values = list(booking_data.values())

    cursor.execute(f'INSERT INTO bookings ({columns}) VALUES ({placeholders})', values)
    booking_id = _db_manager.get_lastrowid(cursor)

    conn.commit()
    conn.close()

    return booking_id


def update_booking(booking_id, booking_data):
    """更新预订信息"""
    conn = _db_manager.get_connection()
    cursor = conn.cursor()

    ph = _db_manager.get_placeholder()

    set_clause = ', '.join([f'{key} = {ph}' for key in booking_data.keys()])
    values = list(booking_data.values()) + [booking_id]

    cursor.execute(f'UPDATE bookings SET {set_clause} WHERE id = {ph}', values)

    conn.commit()
    success = cursor.rowcount > 0
    conn.close()

    return success


def delete_booking(booking_id):
    """删除预订"""
    conn = _db_manager.get_connection()
    cursor = conn.cursor()

    ph = _db_manager.get_placeholder()

    cursor.execute(f'DELETE FROM bookings WHERE id = {ph}', (booking_id,))

    conn.commit()
    success = cursor.rowcount > 0
    conn.close()

    return success


def search_bookings(keyword, limit=100):
    """搜索预订记录"""
    conn = _db_manager.get_connection()
    cursor = conn.cursor()

    ph = _db_manager.get_placeholder()
    like_pattern = f'%{keyword}%'

    cursor.execute(f'''
        SELECT * FROM bookings
        WHERE hotel LIKE {ph} OR country LIKE {ph} OR market_segment LIKE {ph}
        LIMIT {ph}
    ''', (like_pattern, like_pattern, like_pattern, limit))

    rows = cursor.fetchall()
    conn.close()

    if _db_manager.db_type == 'postgresql':
        desc = cursor.description
        col_names = [col[0] for col in desc] if desc else []
        return [dict(zip(col_names, row)) for row in rows]
    else:
        return [dict(row) for row in rows]


def get_statistics():
    """获取统计信息"""
    conn = _db_manager.get_connection()
    cursor = conn.cursor()

    stats = {}

    cursor.execute('SELECT COUNT(*) FROM bookings')
    stats['total_bookings'] = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM bookings WHERE is_canceled = 1')
    stats['canceled_bookings'] = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM bookings WHERE is_canceled = 0')
    stats['confirmed_bookings'] = cursor.fetchone()[0]

    cursor.execute('SELECT AVG(lead_time) FROM bookings')
    stats['avg_lead_time'] = cursor.fetchone()[0]

    cursor.execute('SELECT AVG(adr) FROM bookings')
    stats['avg_adr'] = cursor.fetchone()[0]

    conn.close()

    return stats


# ==================== 用户管理相关函数 ====================

def create_user(username, password, role='staff'):
    """
    创建新用户
    参数:
        username: 用户名（唯一）
        password: 明文密码
        role: 角色 ('staff', 'manager', 或 'admin')
    返回:
        新用户的ID，如果用户名已存在则返回None
    """
    from werkzeug.security import generate_password_hash

    conn = _db_manager.get_connection()
    cursor = conn.cursor()

    ph = _db_manager.get_placeholder()

    try:
        # 检查用户名是否已存在
        cursor.execute(f'SELECT id FROM users WHERE username = {ph}', (username,))
        if cursor.fetchone():
            print(f"错误: 用户名 '{username}' 已存在")
            return None

        # 哈希处理密码
        password_hash = generate_password_hash(password)

        # 验证角色是否合法
        valid_roles = ['staff', 'manager', 'admin']
        if role not in valid_roles:
            raise ValueError(f"无效的角色: {role}，有效角色为: {valid_roles}")

        # 创建用户
        cursor.execute(f'''
            INSERT INTO users (username, password_hash, role)
            VALUES ({ph}, {ph}, {ph})
        ''', (username, password_hash, role))

        user_id = _db_manager.get_lastrowid(cursor)
        conn.commit()

        print(f"成功创建用户: {username} (角色: {role})")
        return user_id

    except Exception as e:
        print(f"创建用户失败: {e}")
        conn.rollback()
        return None
    finally:
        conn.close()


def get_user_by_username(username):
    """根据用户名查询用户"""
    conn = _db_manager.get_connection()
    cursor = conn.cursor()

    ph = _db_manager.get_placeholder()

    cursor.execute(f'SELECT * FROM users WHERE username = {ph}', (username,))
    row = cursor.fetchone()

    conn.close()

    if row is None:
        return None

    if _db_manager.db_type == 'postgresql':
        desc = cursor.description
        col_names = [col[0] for col in desc] if desc else []
        return dict(zip(col_names, row))
    else:
        return dict(row)


def get_user_by_id(user_id):
    """根据ID查询用户"""
    conn = _db_manager.get_connection()
    cursor = conn.cursor()

    ph = _db_manager.get_placeholder()

    cursor.execute(f'SELECT * FROM users WHERE id = {ph}', (user_id,))
    row = cursor.fetchone()

    conn.close()

    if row is None:
        return None

    if _db_manager.db_type == 'postgresql':
        desc = cursor.description
        col_names = [col[0] for col in desc] if desc else []
        return dict(zip(col_names, row))
    else:
        return dict(row)


def get_all_users():
    """获取所有用户列表"""
    conn = _db_manager.get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT id, username, role, is_active, created_at, updated_at
        FROM users
        ORDER BY created_at DESC
    ''')
    rows = cursor.fetchall()

    conn.close()

    if _db_manager.db_type == 'postgresql':
        desc = cursor.description
        col_names = [col[0] for col in desc] if desc else []
        return [dict(zip(col_names, row)) for row in rows]
    else:
        return [dict(row) for row in rows]


def update_user(user_id, user_data):
    """
    更新用户信息
    参数:
        user_id: 用户ID
        user_data: 要更新的字段字典（可包含: username, password, role, is_active）
    返回:
        是否更新成功
    """
    from werkzeug.security import generate_password_hash

    conn = _db_manager.get_connection()
    cursor = conn.cursor()

    ph = _db_manager.get_placeholder()

    try:
        # 构建更新语句
        update_fields = []
        values = []

        if 'username' in user_data:
            update_fields.append(f'username = {ph}')
            values.append(user_data['username'])

        if 'password' in user_data:
            update_fields.append(f'password_hash = {ph}')
            values.append(generate_password_hash(user_data['password']))

        if 'role' in user_data:
            valid_roles = ['staff', 'manager', 'admin']
            if user_data['role'] not in valid_roles:
                raise ValueError(f"无效的角色: {user_data['role']}")
            update_fields.append(f'role = {ph}')
            values.append(user_data['role'])

        if 'is_active' in user_data:
            update_fields.append(f'is_active = {ph}')
            values.append(1 if user_data['is_active'] else 0)

        if not update_fields:
            return False

        # 添加更新时间戳
        update_fields.append('updated_at = CURRENT_TIMESTAMP')
        values.append(user_id)

        # 执行更新
        set_clause = ', '.join(update_fields)
        cursor.execute(f'UPDATE users SET {set_clause} WHERE id = {ph}', values)

        success = cursor.rowcount > 0
        conn.commit()

        if success:
            print(f"成功更新用户 ID: {user_id}")

        return success

    except Exception as e:
        print(f"更新用户失败: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


def delete_user(user_id):
    """
    删除用户
    参数:
        user_id: 用户ID
    返回:
        是否删除成功
    """
    conn = _db_manager.get_connection()
    cursor = conn.cursor()

    ph = _db_manager.get_placeholder()

    try:
        # 不允许删除最后一个管理员
        cursor.execute(f'SELECT role FROM users WHERE id = {ph}', (user_id,))
        user = cursor.fetchone()

        if user:
            role = user[0] if isinstance(user, tuple) else user.get('role')
            if role == 'admin':
                cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
                admin_count = cursor.fetchone()[0]
                if admin_count <= 1:
                    print("错误: 不能删除唯一的管理员账号")
                    return False

        cursor.execute(f'DELETE FROM users WHERE id = {ph}', (user_id,))
        success = cursor.rowcount > 0
        conn.commit()

        if success:
            print(f"成功删除用户 ID: {user_id}")

        return success

    except Exception as e:
        print(f"删除用户失败: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


def verify_password(username, password):
    """
    验证用户密码
    参数:
        username: 用户名
        password: 明文密码
    返回:
        如果验证成功返回用户字典，否则返回None
    """
    from werkzeug.security import check_password_hash

    user = get_user_by_username(username)

    if not user:
        print(f"验证失败: 用户 '{username}' 不存在")
        return None

    if not user.get('is_active'):
        print(f"验证失败: 用户 '{username}' 已被禁用")
        return None

    if check_password_hash(user['password_hash'], password):
        print(f"密码验证成功: {username}")
        # 返回时不包含密码哈希
        user.pop('password_hash', None)
        return user

    print(f"验证失败: 用户 '{username}' 密码错误")
    return None


# ==================== 数据库状态查询函数 ====================

def get_database_status():
    """获取数据库状态信息"""
    conn = _db_manager.get_connection()
    cursor = conn.cursor()

    info = {
        'db_type': _db_manager.db_type,
        'db_info': _db_manager.db_info,
        'tables': {}
    }

    # 获取各表行数
    for table in ['bookings', 'users']:
        try:
            cursor.execute(f'SELECT COUNT(*) FROM {table}')
            info['tables'][table] = cursor.fetchone()[0]
        except Exception:
            info['tables'][table] = 0

    conn.close()
    return info


if __name__ == '__main__':
    # 测试代码
    print(f"数据库类型: {_db_manager.db_type}")
    print(f"数据库信息: {_db_manager.db_info}")
    status = get_database_status()
    print(f"数据库状态: {status}")
