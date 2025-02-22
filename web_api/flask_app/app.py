from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

FASTAPI_URL = "http://127.0.0.1:8000/predict"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        input_data = {
            "gender": request.form["gender"],
            "age": float(request.form["age"]),
            "hypertension": int(request.form["hypertension"]),
            "heart_disease": int(request.form["heart_disease"]),
            "ever_married": request.form["ever_married"],
            "work_type": request.form["work_type"],
            "residence_type": request.form["residence_type"],
            "avg_glucose_level": float(request.form["avg_glucose_level"]),
            "bmi": float(request.form["bmi"]),
            "smoking_status": request.form["smoking_status"],
        }
        
        try:
            response = requests.post(FASTAPI_URL, json=input_data)
            response_data = response.json()
            
            if response.status_code == 200:
                prediction = response_data["prediction"]
                probability = response_data["probability"]
                
                return render_template(
                    "index.html",
                    prediction=prediction,
                    probability=round(probability, 4),
                    input_data=input_data,
                )
            else:
                error_message = response_data.get("detail", "Unknown error occurred.")
                return render_template("index.html", error=error_message, input_data=input_data)
        except Exception as e:
            return render_template("index.html", error=str(e), input_data=input_data)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
