from flask import Flask, request, jsonify
import util

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict_land_price():
    longitude = request.form['longitude']
    latitude = request.form['latitude']
    radius = int(request.form['radius'])
    location = latitude+','+longitude

    response = jsonify(util.get_estimated_price(location=location, radius=radius))

    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


if __name__ == '__main__':
    print("Starting Python Flask server for Colombo Land Price Prediction")
    util.load_saved_artifacts()
    app.run()
