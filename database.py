
# 尝试使用数据库抽象层（支持 SQLite / PostgreSQL 双引擎切换）
try:
    from database_abstract import (
        _db_manager, DatabaseManager, USE_SQLITE, DATABASE_URL,
        init_db as _abstract_init_db,
        init_default_admin as _abstract_init_default_admin,
        import_csv_to_db as _abstract_import_csv_to_db,
        get_all_bookings as _abstract_get_all_bookings,
        get_booking_by_id as _abstract_get_booking_by_id,
        create_booking as _abstract_create_booking,
        update_booking as _abstract_update_booking,
        delete_booking as _abstract_delete_booking,
        search_bookings as _abstract_search_bookings,
        get_statistics as _abstract_get_statistics,
        create_user as _abstract_create_user,
        get_user_by_username as _abstract_get_user_by_username,
        get_user_by_id as _abstract_get_user_by_id,
        get_all_users as _abstract_get_all_users,
        update_user as _abstract_update_user,
        delete_user as _abstract_delete_user,
        verify_password as _abstract_verify_password,
        get_database_status as _abstract_get_database_status
    )
    USING_ABSTRACT_LAYER = True
except ImportError:
    USING_ABSTRACT_LAYER = False

import sqlite3
import pandas as pd
import os
from werkzeug.security import generate_password_hash, check_password_hash

DB_FILE = 'hotel_bookings.db'


# ==================== 数据库初始化函数 ====================

def init_db():
    """初始化数据库，创建所有表"""
    if USING_ABSTRACT_LAYER:
        return _abstract_init_db()

    # 原有 SQLite 逻辑（向后兼容）
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # 创建预订表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bookings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hotel TEXT,
            is_canceled INTEGER,
            lead_time INTEGER,
            arrival_date_year INTEGER,
            arrival_date_month INTEGER,
            arrival_date_week_number INTEGER,
            arrival_date_day_of_month INTEGER,
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
            reservation_status_date TEXT
        )
    ''')

    # 创建用户表（RBAC权限管理）
    init_users_table(cursor)

    conn.commit()
    conn.close()

    # 初始化默认管理员账号
    init_default_admin()


def init_users_table(cursor):
    """创建用户表"""
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('staff', 'manager', 'admin')),
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')


def init_default_admin():
    """初始化默认用户账号（仅在users表为空时）"""
    if USING_ABSTRACT_LAYER:
        return _abstract_init_default_admin()

    # 原有 SQLite 逻辑（向后兼容）
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

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
        cursor.executemany('''
            INSERT INTO users (username, password_hash, role)
            VALUES (?, ?, ?)
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
    if USING_ABSTRACT_LAYER:
        return _abstract_import_csv_to_db(csv_path)

    # 原有 SQLite 逻辑（向后兼容）
    if not os.path.exists(DB_FILE):
        init_db()

    df = pd.read_csv(csv_path)

    conn = sqlite3.connect(DB_FILE)
    df.to_sql('bookings', conn, if_exists='replace', index=False)
    conn.close()

    print(f"成功导入 {len(df)} 条记录到数据库")


def get_all_bookings(limit=100, offset=0):
    """获取所有预订记录（分页）"""
    if USING_ABSTRACT_LAYER:
        return _abstract_get_all_bookings(limit, offset)

    # 原有 SQLite 逻辑（向后兼容）
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM bookings LIMIT ? OFFSET ?', (limit, offset))
    rows = cursor.fetchall()

    cursor.execute('SELECT COUNT(*) FROM bookings')
    total = cursor.fetchone()[0]

    conn.close()

    bookings = [dict(row) for row in rows]
    return bookings, total


def get_booking_by_id(booking_id):
    """根据ID获取预订详情"""
    if USING_ABSTRACT_LAYER:
        return _abstract_get_booking_by_id(booking_id)

    # 原有 SQLite 逻辑（向后兼容）
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM bookings WHERE id = ?', (booking_id,))
    row = cursor.fetchone()

    conn.close()

    return dict(row) if row else None


def create_booking(booking_data):
    """创建新预订"""
    if USING_ABSTRACT_LAYER:
        return _abstract_create_booking(booking_data)

    # 原有 SQLite 逻辑（向后兼容）
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    columns = ', '.join(booking_data.keys())
    placeholders = ', '.join(['?'] * len(booking_data))
    values = list(booking_data.values())

    cursor.execute(f'INSERT INTO bookings ({columns}) VALUES ({placeholders})', values)
    booking_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return booking_id


def update_booking(booking_id, booking_data):
    """更新预订信息"""
    if USING_ABSTRACT_LAYER:
        return _abstract_update_booking(booking_id, booking_data)

    # 原有 SQLite 逻辑（向后兼容）
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    set_clause = ', '.join([f'{key} = ?' for key in booking_data.keys()])
    values = list(booking_data.values()) + [booking_id]

    cursor.execute(f'UPDATE bookings SET {set_clause} WHERE id = ?', values)

    conn.commit()
    conn.close()

    return cursor.rowcount > 0


def delete_booking(booking_id):
    """删除预订"""
    if USING_ABSTRACT_LAYER:
        return _abstract_delete_booking(booking_id)

    # 原有 SQLite 逻辑（向后兼容）
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute('DELETE FROM bookings WHERE id = ?', (booking_id,))

    conn.commit()
    conn.close()

    return cursor.rowcount > 0


def search_bookings(keyword, limit=100):
    """搜索预订记录"""
    if USING_ABSTRACT_LAYER:
        return _abstract_search_bookings(keyword, limit)

    # 原有 SQLite 逻辑（向后兼容）
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute('''
        SELECT * FROM bookings
        WHERE hotel LIKE ? OR country LIKE ? OR market_segment LIKE ?
        LIMIT ?
    ''', (f'%{keyword}%', f'%{keyword}%', f'%{keyword}%', limit))

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_statistics():
    """获取统计信息"""
    if USING_ABSTRACT_LAYER:
        return _abstract_get_statistics()

    # 原有 SQLite 逻辑（向后兼容）
    conn = sqlite3.connect(DB_FILE)
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
    if USING_ABSTRACT_LAYER:
        return _abstract_create_user(username, password, role)

    # 原有 SQLite 逻辑（向后兼容）
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    try:
        # 检查用户名是否已存在
        cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
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
        cursor.execute('''
            INSERT INTO users (username, password_hash, role)
            VALUES (?, ?, ?)
        ''', (username, password_hash, role))

        user_id = cursor.lastrowid
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
    if USING_ABSTRACT_LAYER:
        return _abstract_get_user_by_username(username)

    # 原有 SQLite 逻辑（向后兼容）
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()

    conn.close()

    return dict(user) if user else None


def get_user_by_id(user_id):
    """根据ID查询用户"""
    if USING_ABSTRACT_LAYER:
        return _abstract_get_user_by_id(user_id)

    # 原有 SQLite 逻辑（向后兼容）
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()

    conn.close()

    return dict(user) if user else None


def get_all_users():
    """获取所有用户列表"""
    if USING_ABSTRACT_LAYER:
        return _abstract_get_all_users()

    # 原有 SQLite 逻辑（向后兼容）
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute('''
        SELECT id, username, role, is_active, created_at, updated_at
        FROM users
        ORDER BY created_at DESC
    ''')
    rows = cursor.fetchall()

    conn.close()

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
    if USING_ABSTRACT_LAYER:
        return _abstract_update_user(user_id, user_data)

    # 原有 SQLite 逻辑（向后兼容）
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    try:
        # 构建更新语句
        update_fields = []
        values = []

        if 'username' in user_data:
            update_fields.append('username = ?')
            values.append(user_data['username'])

        if 'password' in user_data:
            update_fields.append('password_hash = ?')
            values.append(generate_password_hash(user_data['password']))

        if 'role' in user_data:
            valid_roles = ['staff', 'manager', 'admin']
            if user_data['role'] not in valid_roles:
                raise ValueError(f"无效的角色: {user_data['role']}")
            update_fields.append('role = ?')
            values.append(user_data['role'])

        if 'is_active' in user_data:
            update_fields.append('is_active = ?')
            values.append(1 if user_data['is_active'] else 0)

        if not update_fields:
            return False

        # 添加更新时间戳
        update_fields.append('updated_at = CURRENT_TIMESTAMP')
        values.append(user_id)

        # 执行更新
        set_clause = ', '.join(update_fields)
        cursor.execute(f'UPDATE users SET {set_clause} WHERE id = ?', values)

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
    if USING_ABSTRACT_LAYER:
        return _abstract_delete_user(user_id)

    # 原有 SQLite 逻辑（向后兼容）
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    try:
        # 不允许删除最后一个管理员
        cursor.execute('SELECT role FROM users WHERE id = ?', (user_id,))
        user = cursor.fetchone()

        if user and user[0] == 'admin':
            cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
            admin_count = cursor.fetchone()[0]
            if admin_count <= 1:
                print("错误: 不能删除唯一的管理员账号")
                return False

        cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
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
    if USING_ABSTRACT_LAYER:
        return _abstract_verify_password(username, password)

    # 原有 SQLite 逻辑（向后兼容）
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


if __name__ == '__main__':
    if not os.path.exists(DB_FILE):
        init_db()
        if os.path.exists('hotel_bookings.csv'):
            import_csv_to_db('hotel_bookings.csv')
