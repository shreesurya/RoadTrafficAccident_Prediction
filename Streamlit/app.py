import joblib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


st.set_page_config(page_title="Accident Severity Prediction App",
                   page_icon="🚧", layout="wide")
                   

#pickle_in = open('rta.pkl', 'rb') 
#model = pickle.load(pickle_in)
model = joblib.load('rta.pkl')

D_day = {'Friday': 0, 'Monday': 1, 'Saturday': 2, 'Sunday': 3, 'Thursday': 4, 'Tuesday': 5, 'Wednesday': 6}
D_age_band = {'18-30': 0, '31-50': 1, 'Over 51': 2, 'Under 18': 3, 'Unknown': 4}
D_lanes_or_medians = {'Double carriageway (median)': 0, 'One way': 1, 'Two-way (divided with broken lines road marking)': 2, 'Two-way (divided with solid lines road marking)': 3, 'Undivided Two way': 4, 'Unknown': 5, 'other': 6}
D_cause_of_accident = {'Changing lane to the left': 0, 'Changing lane to the right': 1, 'Driving at high speed': 2, 'Driving carelessly': 3, 'Driving to the left': 4, 'Driving under the influence of drugs': 5, 'Drunk driving': 6, 'Getting off the vehicle improperly': 7, 'Improper parking': 8, 'Moving Backward': 9, 'No distancing': 10, 'No priority to pedestrian': 11, 'No priority to vehicle': 12, 'Other': 13, 'Overloading': 14, 'Overspeed': 15, 'Overtaking': 16, 'Overturning': 17, 'Turnover': 18, 'Unknown': 19}
D_number_of_casualties = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}
D_types_of_junction= {'Crossing': 0, 'No junction': 1, 'O Shape': 2, 'Other': 3, 'T Shape': 4, 'Unknown': 5, 'X Shape': 6, 'Y Shape': 7}
D_light_conditions = {'Darkness - lights lit': 0, 'Darkness - lights unlit': 1, 'Darkness - no lighting': 2, 'Daylight': 3}
D_number_of_vehicles_involed = {1: 0, 2: 1, 3: 2, 4: 3, 6: 4, 7: 5}

#Accident_Severity = ('serious_injury','slight_injury','fatal_injury') 
options_day = ['Sunday', "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
options_Age_band_of_driver = ['18-30', '31-50', 'Over 51', 'Unknown', 'Under 18']
options_Lanes_or_Medians =['Two-way (divided with broken lines road marking)', 'Undivided Two way','other', 'Double carriageway (median)', 'One way',
       'Two-way (divided with solid lines road marking)']
options_Cause_of_accident = ['Changing lane to the left', 'Changing lane to the right', 'Driving at high speed', 'Driving carelessly', 'Driving to the left', 'Driving under the influence of drugs', 'Drunk driving', 'Getting off the vehicle improperly', 'Improper parking', 'Moving Backward', 'No distancing', 'No priority to pedestrian', 'No priority to vehicle', 'Other', 'Overloading', 'Overspeed', 'Overtaking', 'Overturning', 'Turnover', 'Unknown']
options_Number_of_casualties = ['1', '2', '3', '4', '5', '6', '7', '8']
options_Types_of_Junction= ['Crossing', 'No junction', 'O Shape', 'Other', 'T Shape', 'Unknown', 'X Shape', 'Y Shape']
options_Light_Conditions =['Darkness - lights lit', 'Darkness - lights unlit', 'Darkness - no lighting', 'Daylight']
option_Number_of_Vehicles_Involed =['1','2','3','4','6','7']







features = ['Day_of_week', 'Age_band_of_driver', 'Sex_of_driver','Educational_level', 'Driving_experience','Type_of_vehicle','Owner_of_vehicle', 'Service_year_of_vehicle', 'Area_accident_occured',
       'Lanes_or_Medians', 'Road_allignment', 'Types_of_Junction','Road_surface_type', 'Road_surface_conditions', 'Light_conditions', 'Weather_conditions', 'Type_of_collision',
      'Number_of_vehicles_involved','Number_of_casualties','Vehicle_movement','Casualty_class','Sex_of_casualty','Age_band_of_casualty','Casualty_severity','Pedestrian_movement',
       'Cause_of_accident', 'hour', 'minute']


 

st.markdown("<h1 style='text-align: center;'>Accident Severity Prediction App 🚧</h1>", unsafe_allow_html=True)
def main():

    with st.form('prediction_form'):

        st.header("Enter the input for following features:")


        hour = st.slider("Pickup Hour:", 0, 23, value=0, format="%d")
        minute = st.slider('minute', 0, 59, value=0, format="%d")
        O_Day_of_week = st.selectbox ('Select Day of the Week: ', options=options_day)
        O_Age_band = st.selectbox ('Select Day of the Week: ', options=options_Age_band_of_driver)
        O_Lanes_or_Medians = st.selectbox('Lanes or Medians:',options=options_Lanes_or_Medians)
        O_Cause_of_accident = st.selectbox('Cause of accident:',options=options_Cause_of_accident)
        O_Number_of_casualties = st.slider('Number of casualties', 1, 10, value=0, format="%d")
        O_Types_of_Junction = st.selectbox ('Select Type of Junction: ', options=options_Types_of_Junction)
        O_light_conditions = st.selectbox ('Select Light condition', options=options_Light_Conditions)
        O_Number_of_vehicles_involed = st.slider('Number of vehicles involed', 1, 10, value=0, format="%d")
        submit_values = st.form_submit_button("Predict")

    if submit_values:

        Day_of_week = D_day[O_Day_of_week]
        Age_band_of_driver =  D_age_band[O_Age_band]
        Lanes_or_Medians = D_lanes_or_medians[O_Lanes_or_Medians]
        Cause_of_accident = D_cause_of_accident[O_Cause_of_accident]
        Number_of_casualties = D_number_of_casualties[O_Number_of_casualties]
        Types_of_Junction = D_types_of_junction[O_Types_of_Junction]
        Light_conditions = D_light_conditions[O_light_conditions]
        Number_of_vehicles_involed = D_number_of_vehicles_involed[O_Number_of_vehicles_involed]

        data = np.array([Number_of_casualties, Light_conditions, minute, Number_of_vehicles_involed, Day_of_week, Age_band_of_driver, Types_of_Junction, hour, Cause_of_accident, Lanes_or_Medians]).reshape(1,-1)





        pred = model.predict(data)
    # = get_prediction(data=data, model=model)

        if pred == 0:
            severity = 'seriour_injury' 
        elif pred == 1:
            severity = 'slight_injury'
        else:
            severity = 'fatal_injury' 
        st.write(f"The predicted severity is:  {severity}")

         


     
     
     
if __name__ == '__main__':
    main()      