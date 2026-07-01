from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = float(request.form["age"])
        education_num = float(request.form["education_num"])
        hours_per_week = float(request.form["hours_per_week"])
        capital_gain = float(request.form["capital_gain"])
        capital_loss = float(request.form["capital_loss"])

        # Create DataFrame with the same feature names used during training
        features = pd.DataFrame({
            "age": [age],
            "education-num": [education_num],
            "hours-per-week": [hours_per_week],
            "capital-gain": [capital_gain],
            "capital-loss": [capital_loss]
        })

        prediction = model.predict(features)[0]

        if prediction == 1:
            result = "Income >50K"
        else:
            result = "Income ≤50K"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")


if __name__ == "__main__":
    app.run(debug=True)