from flask import Flask, request, render_template
import os
import pickle
import numpy as np

app = Flask(__name__, template_folder="templates")

# Path to trained model
MODEL_PATH = os.path.join("artifacts", "model.pkl")

# Load trained model + encoder
with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)

# Handle case: model_data might be dict or direct object
if isinstance(model_data, dict):
    model = model_data["model"]
    label_encoder = model_data.get("encoder", None)
else:
    model = model_data
    label_encoder = None   # fallback

# ---------------- HOME ----------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect features from HTML form
        features = [
            float(request.form["N"]),
            float(request.form["P"]),
            float(request.form["K"]),
            float(request.form["temperature"]),
            float(request.form["humidity"]),
            float(request.form["ph"]),
            float(request.form["rainfall"])
        ]
        features = np.array([features])

        # Predict crop
        pred_class = model.predict(features)[0]

        # Fix: handle both int and string outputs
        if isinstance(pred_class, str):
            prediction = pred_class
        elif label_encoder:
            prediction = label_encoder.inverse_transform([int(pred_class)])[0]
        else:
            prediction = str(pred_class)

    except Exception as e:
        prediction = f"Error: {e}"

    return render_template("index.html", prediction_text=f"Recommended Crop: 🌱 {prediction}")


if __name__ == "__main__":
    app.run(debug=True)
