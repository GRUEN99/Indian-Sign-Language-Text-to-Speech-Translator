from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from utils.predict import predict_letter

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return "Flask is running"
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "landmarks" not in data:
        return jsonify({"error": "No landmarks provided"}), 400

    try:
        landmarks = data["landmarks"]
        prediction = predict_letter(landmarks)
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)