from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load ML Model & Encoder
model = joblib.load("model/crop_model.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")

# Dummy values for missing features
DEFAULT_N = 50   # Adjust based on dataset
DEFAULT_P = 50
DEFAULT_K = 50
DEFAULT_HUMIDITY = 60.0
DEFAULT_PH = 6.5

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/crop-suggestion", methods=["POST"])
def suggest_crop():
    try:
        data = request.json
        print("Incoming data:", data)  # Debugging log

        # ✅ Get Inputs
        temperature = float(data.get("temperature"))
        rainfall = float(data.get("rainfall"))

        # ✅ Construct Input DataFrame with missing feature placeholders
        input_data = pd.DataFrame([[
            DEFAULT_N, DEFAULT_P, DEFAULT_K, temperature, DEFAULT_HUMIDITY, DEFAULT_PH, rainfall
        ]], columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])

        # ✅ Make Prediction
        prediction = model.predict(input_data)[0]
        predicted_crop = label_encoder.inverse_transform([prediction])[0]

        return jsonify({"suggested_crop": predicted_crop})

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

