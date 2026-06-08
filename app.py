
from flask import Flask, request, jsonify
from flask_cors import CORS
import database
from prediction_service import get_prediction_service
import os

app = Flask(__name__)
CORS(app)

# 初始化数据库
if not os.path.exists('hotel_bookings.db'):
    database.init_db()
    if os.path.exists('hotel_bookings.csv'):
        database.import_csv_to_db('hotel_bookings.csv')

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'message': 'Hotel Booking System is running'})

# 预订管理API
@app.route('/api/bookings', methods=['GET'])
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

@app.route('/api/bookings/&lt;int:booking_id&gt;', methods=['GET'])
def get_booking(booking_id):
    booking = database.get_booking_by_id(booking_id)
    if booking:
        return jsonify(booking)
    return jsonify({'error': 'Booking not found'}), 404

@app.route('/api/bookings', methods=['POST'])
def create_booking():
    booking_data = request.json
    booking_id = database.create_booking(booking_data)
    return jsonify({'id': booking_id, 'message': 'Booking created successfully'}), 201

@app.route('/api/bookings/&lt;int:booking_id&gt;', methods=['PUT'])
def update_booking(booking_id):
    booking_data = request.json
    success = database.update_booking(booking_id, booking_data)
    if success:
        return jsonify({'message': 'Booking updated successfully'})
    return jsonify({'error': 'Booking not found'}), 404

@app.route('/api/bookings/&lt;int:booking_id&gt;', methods=['DELETE'])
def delete_booking(booking_id):
    success = database.delete_booking(booking_id)
    if success:
        return jsonify({'message': 'Booking deleted successfully'})
    return jsonify({'error': 'Booking not found'}), 404

# 统计API
@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    stats = database.get_statistics()
    return jsonify(stats)

# 预测API
@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    booking_data = data.get('booking', {})
    model_name = data.get('model', 'Random Forest')
    
    try:
        service = get_prediction_service()
        result = service.predict(booking_data, model_name)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/all', methods=['POST'])
def predict_all():
    data = request.json
    booking_data = data.get('booking', {})
    
    try:
        service = get_prediction_service()
        results = service.predict_all_models(booking_data)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    try:
        service = get_prediction_service()
        models = service.get_available_models()
        return jsonify({'models': models})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/performance', methods=['GET'])
def get_model_performance():
    try:
        service = get_prediction_service()
        performance = service.get_model_performance()
        if performance:
            return jsonify(performance)
        return jsonify({'message': 'No performance data available'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("启动酒店预订智能管理系统...")
    app.run(debug=True, host='0.0.0.0', port=5000)

