from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/hello')
def hello():
    return "Hello"


if __name__ == '__main__':
    print("Starting Python Flask server for Colombo Land Price Prediction")
    app.run()
