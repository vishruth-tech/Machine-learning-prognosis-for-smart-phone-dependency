import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load pre-trained models
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
et_model = pickle.load(open(os.path.join(MODEL_DIR, "et_model.pkl"), "rb"))
stacking_model = pickle.load(open(os.path.join(MODEL_DIR, "stacking.pkl"), "rb"))
catboost = pickle.load(open(os.path.join(MODEL_DIR, "catboost.pkl"), "rb"))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/login")
def login():
    return render_template("login.html")


@app.route("/upload")
def upload():
    return render_template("upload.html")


@app.route("/preview", methods=["POST"])
def preview():
    if request.method == "POST":
        dataset = request.files.get("datasetfile")
        if dataset:
            df = pd.read_csv(dataset)
            return render_template("preview.html", df_view=df)
        return render_template("upload.html", error="Please upload a valid CSV file.")


@app.route("/prediction")
def prediction():
    return render_template("prediction.html")


@app.route("/performance")
def performance():
    return render_template("performance.html")


@app.route("/chart")
def chart():
    return render_template("chart.html")


@app.route("/result")
def result():
    return render_template("result.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            # 18 Features matching model training schema
            gender = request.form.get("Gender", "0")
            pictures_of_notes = request.form.get("pictures_of_notes", "0")
            buy_books = request.form.get("buy_books", "0")
            run_for_charger = request.form.get("run_for_charger", "0")
            worry_about_phone = request.form.get("worry_about_losing_phone", "0")
            phone_in_bathroom = request.form.get("phone_in_bathroom", "0")
            phone_in_social = request.form.get("phone_in_social_gathering", "0")
            check_without_notif = request.form.get("check_phone_without_notification", "0")
            phone_before_sleep = request.form.get("phone_before_sleep", "0")
            phone_next_to_you = request.form.get("phone_next_to_you", "0")
            missed_calls = request.form.get("missed calls", "0")
            yourself_relying = request.form.get("yourself relying", "0")
            watching_tv = request.form.get("watching TV", "0")
            panic_attack = request.form.get("panic attack ", "0")
            messages_or_checking = request.form.get("messages or checking ", "0")
            playing_games = request.form.get("playing games", "0")
            without_phone = request.form.get("without", "0")
            battery = request.form.get("battery", "0")

            selected_model = request.form.get("model", "Extra Tree")

            feature_list = [
                float(gender),
                float(pictures_of_notes),
                float(buy_books),
                float(run_for_charger),
                float(worry_about_phone),
                float(phone_in_bathroom),
                float(phone_in_social),
                float(check_without_notif),
                float(phone_before_sleep),
                float(phone_next_to_you),
                float(missed_calls),
                float(yourself_relying),
                float(watching_tv),
                float(panic_attack),
                float(messages_or_checking),
                float(playing_games),
                float(without_phone),
                float(battery),
            ]

            final_features = np.array([feature_list])

            # Select model
            if selected_model == "CatBoost":
                raw_pred = catboost.predict(final_features)
                model_name = "CatBoost Classifier"
            elif selected_model == "Stacking":
                raw_pred = stacking_model.predict(final_features)
                model_name = "Stacking Classifier"
            else:
                raw_pred = et_model.predict(final_features)
                model_name = "Extra Trees Classifier"

            val = raw_pred[0]
            if isinstance(val, (np.ndarray, list)):
                val = val[0]

            prediction_label = "Yes" if int(round(float(val))) == 1 else "No"

            return render_template(
                "result.html",
                model=model_name,
                prediction_text=prediction_label,
                Gender=gender,
                For_how_long_do_you_use_your_phone_for_playing_games=playing_games,
                Do_you_use_your_phone_to_click_pictures_of_class_notes=pictures_of_notes,
                Do_you_buy_books_access_books_from_your_mobile=buy_books,
                When_your_phones_battery_dies_out_do_you_run_for_the_charger=run_for_charger,
                Do_you_worry_about_losing_your_cell_phone=worry_about_phone,
                Do_you_take_your_phone_to_the_bathroom=phone_in_bathroom,
                Do_you_use_your_phone_in_any_social_gathering_parties=phone_in_social,
                Do_you_often_check_your_phone_without_any_notification=check_without_notif,
                Do_you_check_your_phone_just_before_going_to_sleep_just_after_waking_up=phone_before_sleep,
                Do_you_keep_your_phone_right_next_to_you_while_sleeping=phone_next_to_you,
                Do_you_check_emails_missed_calls_texts_during_class_time=missed_calls,
                Do_you_find_yourself_relying_on_your_phone_when_things_get_awkward=yourself_relying,
                Are_you_on_your_phone_while_watching_TV_or_eating_food=watching_tv,
                Do_you_have_a_panic_attack_if_you_leave_your_phone_elsewhere=panic_attack,
                You_dont_mind_responding_to_messages_or_checking_your_phone_while_on_date=messages_or_checking,
                Can_you_live_a_day_without_phone=without_phone,
                Does_your_phones_battery_last_a_day=battery,
            )
        except Exception as e:
            return render_template("prediction.html", error=f"Error processing prediction: {e}")


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    app.run(debug=debug_mode, host="0.0.0.0", port=5000)
