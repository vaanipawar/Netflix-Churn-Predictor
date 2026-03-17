from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form["age"])
        gender = request.form["gender"]
        country = request.form["country"]
        subscription_type = request.form["subscription_type"]
        payment_method = request.form["payment_method"]
        primary_device = request.form["primary_device"]
        account_age_months = int(request.form["account_age_months"])
        monthly_fee = int(request.form["monthly_fee"])
        devices_used = int(request.form["devices_used"])
        favorite_genre = request.form["favorite_genre"]
        avg_watch_time_minutes = int(request.form["avg_watch_time_minutes"])
        watch_sessions_per_week = int(request.form["watch_sessions_per_week"])
        binge_watch_sessions = int(request.form["binge_watch_sessions"])
        completion_rate = int(request.form["completion_rate"])
        rating_given = int(request.form["rating_given"])
        content_interactions = int(request.form["content_interactions"])
        recommendation_click_rate = int(request.form["recommendation_click_rate"])
        days_since_last_login = int(request.form["days_since_last_login"])

        X_new = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "country": country,
            "subscription_type": subscription_type,
            "payment_method": payment_method,
            "primary_device": primary_device,
            "account_age_months": account_age_months,
            "monthly_fee": monthly_fee,
            "devices_used": devices_used,
            "favorite_genre": favorite_genre,
            "avg_watch_time_minutes": avg_watch_time_minutes,
            "watch_sessions_per_week": watch_sessions_per_week,
            "binge_watch_sessions": binge_watch_sessions,
            "completion_rate": completion_rate,
            "rating_given": rating_given,
            "content_interactions": content_interactions,
            "recommendation_click_rate": recommendation_click_rate,
            "days_since_last_login": days_since_last_login
        }])
        prediction = int(model.predict(X_new)[0])
        label = "Churned " if prediction == 1 else "Not Churned "
        return render_template("index.html", result=f"Prediction: {label}")


if __name__ == "__main__":
    app.run(debug=True)

