from flask import Flask,render_template,url_for,request

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import numpy as np 


import pickle

app = Flask(__name__)
et_model= pickle.load(open('model/et_model.pkl','rb'))
stacking_model = pickle.load(open('model/stacking.pkl','rb'))
catboost = pickle.load(open('model/catboost.pkl','rb'))


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/login')
def login():
    return render_template("login.html")


@app.route('/upload')
def upload():
    return render_template("upload.html")

@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset)
        return render_template("preview.html",df_view = df)


@app.route('/prediction')
def prediction():
    return render_template("prediction.html")

@app.route('/result')
def result():
    return render_template("result.html")

@app.route('/predict',methods=["POST"])
def predict():
    if request.method == 'POST':
             Gender = request.form['Gender']
             Do_you_use_your_phone_to_click_pictures_of_class_notes = request.form['pictures_of_notes']
             Do_you_buy_books_access_books_from_your_mobile = request.form['buy_books']
             When_your_phones_battery_dies_out_do_you_run_for_the_charger = request.form['run_for_charger']
             Do_you_worry_about_losing_your_cell_phone = request.form['worry_about_losing_phone']
             Do_you_take_your_phone_to_the_bathroom = request.form['phone_in_bathroom']
             Do_you_use_your_phone_in_any_social_gathering_parties = request.form['phone_in_social_gathering']
             Do_you_often_check_your_phone_without_any_notification = request.form['check_phone_without_notification']
             Do_you_check_your_phone_just_before_going_to_sleep_just_after_waking_up = request.form['phone_before_sleep']
             Do_you_keep_your_phone_right_next_to_you_while_sleeping = request.form['phone_next_to_you']
             Do_you_check_emails_missed_calls_texts_during_class_time = request.form['missed calls']
             Do_you_find_yourself_relying_on_your_phone_when_things_get_awkward = request.form['yourself relying']
             Are_you_on_your_phone_while_watching_TV_or_eating_food = request.form['watching TV']
             Do_you_have_a_panic_attack_if_you_leave_your_phone_elsewhere = request.form['panic attack ']
             You_dont_mind_responding_to_messages_or_checking_your_phone_while_on_date = request.form['messages or checking ']
             For_how_long_do_you_use_your_phone_for_playing_games = request.form['playing games']
             Can_you_live_a_day_without_phone = request.form['without']
            
             Does_your_phones_battery_last_a_day = request.form['battery']


             model_name = request.form['model']
             sample_data = [
                  Gender,
                  Do_you_use_your_phone_to_click_pictures_of_class_notes,
    Do_you_buy_books_access_books_from_your_mobile,
    When_your_phones_battery_dies_out_do_you_run_for_the_charger,
    Do_you_worry_about_losing_your_cell_phone,
    Do_you_take_your_phone_to_the_bathroom,
    Do_you_use_your_phone_in_any_social_gathering_parties,
    Do_you_often_check_your_phone_without_any_notification,
    Do_you_check_your_phone_just_before_going_to_sleep_just_after_waking_up,
    Do_you_keep_your_phone_right_next_to_you_while_sleeping,
    Do_you_check_emails_missed_calls_texts_during_class_time,
    Do_you_find_yourself_relying_on_your_phone_when_things_get_awkward,
    Are_you_on_your_phone_while_watching_TV_or_eating_food,
    Do_you_have_a_panic_attack_if_you_leave_your_phone_elsewhere,
    You_dont_mind_responding_to_messages_or_checking_your_phone_while_on_date,
    For_how_long_do_you_use_your_phone_for_playing_games,
    Can_you_live_a_day_without_phone,
       Does_your_phones_battery_last_a_day
        ]

            
        
		# Clean the data by convert from unicode to float 
        
        # clean_data = [float(i) for i in sample_data]
        # int_feature = [x for x in sample_data]
             int_feature = [float(i) for i in sample_data]

            
             ex1 = np.array(int_feature).reshape(1,-1)
		# ex1 = np.array([6.2,3.4,5.4,2.3]).reshape(1,-1)

        # Reloading the Model
             if model_name == 'Extra Tree':
                  result_prediction = et_model.predict(ex1)
                  result_prediction_text = "No" if result_prediction == 0 else "Yes"
           
            
            
             elif model_name == 'Stacking':
                 result_prediction = stacking_model.predict(ex1)
                 result_prediction_text = "No" if result_prediction == 0 else "Yes"

             elif model_name == 'CatBoost':
                 result_prediction = catboost.predict(ex1)
                 result_prediction_text = "No" if result_prediction == 0 else "Yes"

             else:
                return "Invalid model selected."
           
           
        
             model_name_text = model_name
             Gender_text = "Male" if Gender == 1 else "Female"

             Do_you_use_your_phone_to_click_pictures_of_class_notes_text = "Yes" if Do_you_use_your_phone_to_click_pictures_of_class_notes == 1 else "No"
             Do_you_buy_books_access_books_from_your_mobile_text = "Yes" if Do_you_buy_books_access_books_from_your_mobile == 'Yes' else "No"
             When_your_phones_battery_dies_out_do_you_run_for_the_charger_text = "Yes" if When_your_phones_battery_dies_out_do_you_run_for_the_charger == 1 else "No"
             Do_you_worry_about_losing_your_cell_phone_text = "Yes" if Do_you_worry_about_losing_your_cell_phone == 1 else "No"
             Do_you_take_your_phone_to_the_bathroom_text = "Yes" if Do_you_take_your_phone_to_the_bathroom == 1 else "No"
             Do_you_use_your_phone_in_any_social_gathering_parties_text = "Yes" if Do_you_use_your_phone_in_any_social_gathering_parties == 1 else "No"
             Do_you_often_check_your_phone_without_any_notification_text = "Yes" if Do_you_often_check_your_phone_without_any_notification == 1 else "No"
             Do_you_check_your_phone_just_before_going_to_sleep_just_after_waking_up_text = "Yes" if Do_you_check_your_phone_just_before_going_to_sleep_just_after_waking_up == 1 else "No"
             Do_you_keep_your_phone_right_next_to_you_while_sleeping_text = "Yes" if Do_you_keep_your_phone_right_next_to_you_while_sleeping == 1 else "No"
             Do_you_check_emails_missed_calls_texts_during_class_time_text = "Yes" if Do_you_check_emails_missed_calls_texts_during_class_time == 1 else "No"
             Do_you_find_yourself_relying_on_your_phone_when_things_get_awkward_text = "Yes" if Do_you_find_yourself_relying_on_your_phone_when_things_get_awkward == 1 else "No"
             Are_you_on_your_phone_while_watching_TV_or_eating_food_text = "Yes" if Are_you_on_your_phone_while_watching_TV_or_eating_food == 1 else "No"
             Do_you_have_a_panic_attack_if_you_leave_your_phone_elsewhere_text = "Yes" if Do_you_have_a_panic_attack_if_you_leave_your_phone_elsewhere == 1 else "No"
             You_dont_mind_responding_to_messages_or_checking_your_phone_while_on_date_text = "Yes" if You_dont_mind_responding_to_messages_or_checking_your_phone_while_on_date == 1 else "No"
             For_how_long_do_you_use_your_phone_for_playing_games_text = "Less than 2 hours" if For_how_long_do_you_use_your_phone_for_playing_games == '1' else "More than 2 hours"
             Can_you_live_a_day_without_phone_text = "Yes" if Can_you_live_a_day_without_phone == 1 else "No"
             Does_your_phones_battery_last_a_day_text = "Yes" if Does_your_phones_battery_last_a_day == 1 else "No"
          

   
    return render_template('result.html', prediction_text= result_prediction_text,result_prediction = int(result_prediction), model = model_name,Gender = Gender,
Do_you_use_your_phone_to_click_pictures_of_class_notes = Do_you_use_your_phone_to_click_pictures_of_class_notes,
Do_you_buy_books_access_books_from_your_mobile = Do_you_buy_books_access_books_from_your_mobile,
When_your_phones_battery_dies_out_do_you_run_for_the_charger = When_your_phones_battery_dies_out_do_you_run_for_the_charger,
Do_you_worry_about_losing_your_cell_phone = Do_you_worry_about_losing_your_cell_phone,
Do_you_take_your_phone_to_the_bathroom = Do_you_take_your_phone_to_the_bathroom,
Do_you_use_your_phone_in_any_social_gathering_parties = Do_you_use_your_phone_in_any_social_gathering_parties,
Do_you_often_check_your_phone_without_any_notification = Do_you_often_check_your_phone_without_any_notification,
Do_you_check_your_phone_just_before_going_to_sleep_just_after_waking_up = Do_you_check_your_phone_just_before_going_to_sleep_just_after_waking_up,
Do_you_keep_your_phone_right_next_to_you_while_sleeping = Do_you_keep_your_phone_right_next_to_you_while_sleeping,
Do_you_check_emails_missed_calls_texts_during_class_time = Do_you_check_emails_missed_calls_texts_during_class_time,
Do_you_find_yourself_relying_on_your_phone_when_things_get_awkward = Do_you_find_yourself_relying_on_your_phone_when_things_get_awkward,
Are_you_on_your_phone_while_watching_TV_or_eating_food = Are_you_on_your_phone_while_watching_TV_or_eating_food,
Do_you_have_a_panic_attack_if_you_leave_your_phone_elsewhere = Do_you_have_a_panic_attack_if_you_leave_your_phone_elsewhere,
You_dont_mind_responding_to_messages_or_checking_your_phone_while_on_date = You_dont_mind_responding_to_messages_or_checking_your_phone_while_on_date,
For_how_long_do_you_use_your_phone_for_playing_games = For_how_long_do_you_use_your_phone_for_playing_games,
Can_you_live_a_day_without_phone = Can_you_live_a_day_without_phone,
Does_your_phones_battery_last_a_day = Does_your_phones_battery_last_a_day
)

@app.route('/performance')
def performance():
    return render_template("performance.html")

@app.route('/chart')
def chart():
    return render_template("chart.html")    

if __name__ == '__main__':
	app.run(debug=True)
