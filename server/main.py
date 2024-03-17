from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import util

app = Flask(__name__)
CORS(app)


@app.errorhandler(Exception)
def handle_error(e):
    response = jsonify({'error': str(e)})
    response.status_code = 500
    return response


@app.route('/predict', methods=['POST', 'GET', 'OPTIONS'])
@cross_origin()
def predict_land_price():
    try:
        util.load_saved_artifacts()
        request_data = request.json

        longitude = str(request_data['longitude'])
        latitude = str(request_data['latitude'])
        radius = int(request_data['radius'])
        land_type = request_data['landType']
        location = latitude + ',' + longitude

        print(location + ' ' + radius + ' '+ land_type)
        msg = {
            'message': "Request received"
        }
        response = jsonify(msg)
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'

        return response

    except Exception as e:
        print(e)
        return handle_error(e)


if __name__ == '__main__':
    print("Starting Python Flask server for Colombo Land Price Prediction")
    app.run()
