import os
from flask import Flask, render_template, request

import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

et_model = pickle.load(open('model/et_model.pkl', 'rb'))
stacking_model = pickle.load(open('model/stacking.pkl', 'rb'))
catboost = pickle.load(open('model/catboost.pkl', 'rb'))


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/login')
def login():
    return render_template("login.html")


@app.route('/upload')
def upload():
    return render_template("upload.html")


@app.route('/preview', methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset)
        return render_template("preview.html", df_view=df)


@app.route('/prediction')
def prediction():
    return render_template("prediction.html")


@app.route('/result')
def result():
    return render_template("result.html")


@app.route('/predict', methods=["POST"])
def predict():
    if request.method == 'POST':
        try:
            Gender = request.form['Gender']
            Do_you_use_your_phone_to_click_pictures_of_class_notes = request.form['pictures_of_notes']
            Do_you_buy_books_access_books_from_your_mobile = request.form['buy_books']
            When_your_phones_battery_dies_out_do_you_run_for_the_charger = request.form['run_for_charger']
            Do_you_worry_about_losing_your_cell_phone = request.form['worry_about_losing_phone']
            Do_you_take_your_phone_to_the_bathroom = request.form['phone_in_bathroom']
            Do_you_use_your_phone_in_any_social_gathering_parties = request.form['phone_in_social_gathering']
            Do_you_often_check_your_phone_without_any_notification = request.form['check_phone_without_notification']

            features = [float(i) for i in [
                Gender,
                Do_you_use_your_phone_to_click_pictures_of_class_notes,
                Do_you_buy_books_access_books_from_your_mobile,
                When_your_phones_battery_dies_out_do_you_run_for_the_charger,
                Do_you_worry_about_losing_your_cell_phone,
                Do_you_take_your_phone_to_the_bathroom,
                Do_you_use_your_phone_in_any_social_gathering_parties,
                Do_you_often_check_your_phone_without_any_notification
            ]]

            final_features = [np.array(features)]

            et_prediction = et_model.predict(final_features)
            stacking_prediction = stacking_model.predict(final_features)
            catboost_prediction = catboost.predict(final_features)

            et_output = round(et_prediction[0], 2)
            stacking_output = round(stacking_prediction[0], 2)
            catboost_output = round(catboost_prediction[0], 2)

            return render_template(
                'result.html',
                et_prediction_text=et_output,
                stacking_prediction_text=stacking_output,
                catboost_prediction_text=catboost_output
            )
        except (ValueError, KeyError) as e:
            return render_template('prediction.html', error="Invalid input. Please fill all fields correctly.")


if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug_mode)
