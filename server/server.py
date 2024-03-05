from flask import Flask, request, jsonify
import util

app = Flask(__name__)




if __name__ == '__main__':
    print("Starting Python Flask server for Colombo Land Price Prediction")
    util.load_saved_artifacts()
    app.run()
