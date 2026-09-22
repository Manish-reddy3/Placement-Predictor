from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Your trained sklearn Pipeline (preprocessing + model)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "moddd.pkl")

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Keep the same feature order used during training:
        # IQ, CGPA, 10th marks, 12th marks, BTech CGPA
        iq = float(request.form["iq"])
        cgpa = float(request.form["cgpa"])
        marks_10th = float(request.form["marks_10th"])
        marks_12th = float(request.form["marks_12th"])
        btech_cgpa = float(request.form["btech_cgpa"])

        features = np.array([[iq, cgpa, marks_10th, marks_12th, btech_cgpa]])

        prediction = model.predict(features)[0]

        output = "Placed" if int(prediction) == 1 else "Not Placed"

        return render_template(
            "index.html",
            prediction_text=f"Prediction: {output}"
        )

    except (ValueError, TypeError, KeyError):
        return render_template(
            "index.html",
            prediction_text="Please enter valid numeric values."
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
