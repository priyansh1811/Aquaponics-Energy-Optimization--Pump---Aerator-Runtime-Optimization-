from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models
rf_aerator = joblib.load(os.path.join(BASE_DIR, "rf_aerator_model.pkl"))
rf_pump = joblib.load(os.path.join(BASE_DIR, "rf_pump_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

@app.route("/")
def home():
    return jsonify({"status": "✅ Aquaponics AI API is running!"})

@app.route("/predict_aerator", methods=["POST"])
def predict_aerator():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "❌ No data received."}), 400

        # Define expected feature order (Same as training script)
        feature_order = ["Temperature", "Turbidity", "Dissolved Oxygen", "PH", 
                         "Ammonia", "Nitrate", "Fish Population", "Fish Length", 
                         "Fish Weight", "Energy Cost"]

        # Check for missing features
        missing_features = [feature for feature in feature_order if feature not in data]
        if missing_features:
            return jsonify({"error": f"❌ Missing features: {missing_features}"}), 400

        # Convert input data into a NumPy array
        input_features = np.array([data[feature] for feature in feature_order]).reshape(1, -1)

        # ❌ DO NOT SCALE AERATOR MODEL INPUT (Because it was trained without scaling)
        prediction = rf_aerator.predict(input_features)[0]

        return jsonify({
            "status": "✅ Success",
            "aerator_status": int(prediction),
            "message": "Aerator is ON" if prediction == 1 else "Aerator is OFF"
        })

    except Exception as e:
        return jsonify({"error": f"❌ Internal Server Error: {str(e)}"}), 500

@app.route("/predict_pump", methods=["POST"])
def predict_pump():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "❌ No data received."}), 400

        feature_order = ["Temperature", "Turbidity", "Dissolved Oxygen", "PH", 
                         "Ammonia", "Nitrate", "Fish Population", "Fish Length", 
                         "Fish Weight", "Energy Cost"]

        missing_features = [feature for feature in feature_order if feature not in data]
        if missing_features:
            return jsonify({"error": f"❌ Missing features: {missing_features}"}), 400

        input_features = np.array([data[feature] for feature in feature_order]).reshape(1, -1)

        # ✅ Scale input for Pump Model (Pump Model was trained with StandardScaler)
        input_scaled = scaler.transform(input_features)

        prediction = rf_pump.predict(input_scaled)[0]

        return jsonify({
            "status": "✅ Success",
            "pump_runtime": round(prediction, 2),
            "message": f"Recommended pump runtime: {round(prediction, 2)} min/hr"
        })

    except Exception as e:
        return jsonify({"error": f"❌ Internal Server Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)