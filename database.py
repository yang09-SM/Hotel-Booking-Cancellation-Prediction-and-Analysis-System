
import sqlite3
import pandas as pd
import os

DB_FILE = 'hotel_bookings.db'

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
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
    
    conn.commit()
    conn.close()

def import_csv_to_db(csv_path):
    if not os.path.exists(DB_FILE):
        init_db()
    
    df = pd.read_csv(csv_path)
    
    conn = sqlite3.connect(DB_FILE)
    df.to_sql('bookings', conn, if_exists='replace', index=False)
    conn.close()
    
    print(f"成功导入 {len(df)} 条记录到数据库")

def get_all_bookings(limit=100, offset=0):
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
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM bookings WHERE id = ?', (booking_id,))
    row = cursor.fetchone()
    
    conn.close()
    
    return dict(row) if row else None

def create_booking(booking_data):
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
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    set_clause = ', '.join([f'{key} = ?' for key in booking_data.keys()])
    values = list(booking_data.values()) + [booking_id]
    
    cursor.execute(f'UPDATE bookings SET {set_clause} WHERE id = ?', values)
    
    conn.commit()
    conn.close()
    
    return cursor.rowcount &gt; 0

def delete_booking(booking_id):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM bookings WHERE id = ?', (booking_id,))
    
    conn.commit()
    conn.close()
    
    return cursor.rowcount &gt; 0

def search_bookings(keyword, limit=100):
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

if __name__ == '__main__':
    if not os.path.exists(DB_FILE):
        init_db()
        if os.path.exists('hotel_bookings.csv'):
            import_csv_to_db('hotel_bookings.csv')

