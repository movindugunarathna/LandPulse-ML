from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import os
from dotenv import load_dotenv

import util

load_dotenv()
app = Flask(__name__)
PORT=os.environ.get("PORT")
CORS(app)


@app.errorhandler(Exception)
def handle_error(e):
    response = jsonify({'error': str(e)})
    response.status_code = 500
    return response

@app.route('/')
def hello_landpulse():
    return 'Hello, Landpulse'


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

        predicted_details = util.convert_int_to_str(
        util.get_estimated_price(location=location, land_type=land_type, radius=radius))
        print("Predicted price:" + str(predicted_details))
        response = jsonify(predicted_details)
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'

        return response

    except Exception as e:
        print(e)
        return handle_error(e)


if __name__ == '__main__':
    print("Starting Python Flask server for Colombo Land Price Prediction: "+str(PORT))
    app.run(host="0.0.0.0", port=int(PORT))
