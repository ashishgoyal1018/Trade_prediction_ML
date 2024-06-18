# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import streamlit as st
import pickle
from sklearn.preprocessing import MinMaxScaler

#import os
#os.getcwd()

#os.chdir('D:\\ASTA_Trading\\Trade_experience\\9_Trading_model_Log_reg')
#loaded_model = pickle.load('trained_model.sav','rb')
try:
    with open('trained_model_dt.sav', 'rb') as file:
        loaded_model= pickle.load(file)
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_loaded = False

def trade_prediction(input_data):
# changing the input_data to numpy array
    try:
        input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

        prediction = loaded_model.predict(input_data_reshaped)
        print(prediction)

        if (prediction[0] == 0):
            return'The trade has more than 75% chances of hitting the target'
        else:
            return'The trade has more than 75% chances of hitting the stoploss'
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

def main():
    st.title('Chart Pattern Profitability Prediction')
    
    Date_2= st.number_input('Enter Date of trade',min_value=1,max_value=31,value=15)
    
    dropdown_options_1= ["Mon","Tue","Wed","Thur","Fri"]
    Day= st.selectbox("Select the day:", dropdown_options_1)
    Day_encoded_= {"Mon":0.75, "Tue":0.5, "Wed":0.25, "Thur":0, "Fri":1}
    Day_encoded= Day_encoded_[Day]

    dropdown_options_2= ["BankNifty", "Nifty"]
    Symbol= st.selectbox("Select the Index:", dropdown_options_2)
    Symbol_encoded_= {"BankNifty":0, "Nifty":1}
    Symbol_encoded= Symbol_encoded_[Symbol]
    
    dropdown_options_3= ["3min", "5min","10min","15min"]
    Time_Frame= st.selectbox("Select the Timeframe:", dropdown_options_3)
    Time_Frame_encoded_= {"15min":0, "10min":0.33,"5min":0.66,"3min":1}
    Time_Frame_encoded= Time_Frame_encoded_[Time_Frame]
    
    dropdown_options_4= ["9","10","11","12","13","14","15"]
    Time_of_BC_candle_hr_2= st.selectbox("Time of BC candle: Hour:", dropdown_options_4)
    Time_of_BC_candle_min_2= st.number_input("Time of BC candle: Minutes",min_value=0,max_value=59,value=30)

    dropdown_options_5= ["Yes", "No"]
    Distribution_0_yes_1_no= st.selectbox("Whether setup as per Chart pattern", dropdown_options_5)
    Distribution_0_yes_1_no_encoded_= {"Yes":0, "No":1}
    Distribution_0_yes_1_no_encoded= Distribution_0_yes_1_no_encoded_[Distribution_0_yes_1_no]

    India_Vix_2= st.number_input('Enter the India Vix at time of trade',min_value=13,max_value=26,value=19)
    
    dropdown_options_6= ["Bullish", "Bearish","Neutral"]
    Trend_Day_Chart_1_Bullish_0_Bearish= st.selectbox("Trend in Day chart", dropdown_options_6)
    Trend_Day_Chart_1_Bullish_0_Bearish_encoded_= {"Bullish":1, "Bearish":0, "Neutral":0.5}
    Trend_Day_Chart_1_Bullish_0_Bearish_encoded= Trend_Day_Chart_1_Bullish_0_Bearish_encoded_[Trend_Day_Chart_1_Bullish_0_Bearish]

    dropdown_options_7= ["Bullish", "Bearish","Neutral"]
    Trend_1hr_Chart_1_Bullish_0_Bearish= st.selectbox("Trend in 1hr chart", dropdown_options_7)
    Trend_1hr_Chart_1_Bullish_0_Bearish_encoded_= {"Bullish":1, "Bearish":0, "Neutral":0.5}
    Trend_1hr_Chart_1_Bullish_0_Bearish_encoded= Trend_Day_Chart_1_Bullish_0_Bearish_encoded_[Trend_1hr_Chart_1_Bullish_0_Bearish]

    dropdown_options_8= ["Gap Up", "Gap Down","Flat"]
    Gap_up_0_down_1= st.selectbox("Select the Index Opening", dropdown_options_8)
    Gap_up_0_down_1_encoded_= {"Gap Up":1, "Gap Down":0, "Flat":0.5}
    Gap_up_0_down_1_encoded= Gap_up_0_down_1_encoded_[Gap_up_0_down_1]

    dropdown_options_9= ["Yes", "No"]
    Twice_high_volume_0_Yes_1_No= st.selectbox("Is the BC volume double the next candle volume", dropdown_options_9)
    Twice_high_volume_0_Yes_1_No_encoded_= {"Yes":0, "No":1}
    Twice_high_volume_0_Yes_1_No_encoded= Twice_high_volume_0_Yes_1_No_encoded_[Twice_high_volume_0_Yes_1_No]

    #    def load_data_scale():
    data=pd.read_csv("5_Model_log_reg_Buying_Climax_14062024_scaled.csv")
  #  numerical_cols = ['Date', 'Time_of_BC_candle_hr','Time_of_BC_candle_min', 'India_Vix']
        
  #  scaler_minmax = MinMaxScaler()
 #  data[numerical_cols] = scaler_minmax.fit_transform(data[numerical_cols])
   # scaler_minmax.fit(data[numerical_cols]) 
#        return data

    numerical_cols = ['Date', 'Time_of_BC_candle_hr', 'Time_of_BC_candle_min', 'India_Vix']
    scaler_minmax = MinMaxScaler()
    scaler_minmax.fit(data[numerical_cols])  # Fit the scaler with the training data

    # Prepare input data for scaling
    input_data_unscaled = [[Date_2, int(Time_of_BC_candle_hr_2), Time_of_BC_candle_min_2, India_Vix_2]]
    input_data_scaled = scaler_minmax.transform(input_data_unscaled)


    Predict = ''
    input_data = [round(input_data_scaled[0][0],2), Day_encoded, Symbol_encoded, Time_Frame_encoded,round(input_data_scaled[0][1],2), round(input_data_scaled[0][2],2),Distribution_0_yes_1_no_encoded,round(input_data_scaled[0][3],2), Trend_Day_Chart_1_Bullish_0_Bearish_encoded,Trend_1hr_Chart_1_Bullish_0_Bearish_encoded, Gap_up_0_down_1_encoded,Twice_high_volume_0_Yes_1_No_encoded]

    st.write(f"Input data: {input_data}")

    if st.button('Trade prediction result'):
        prediction_result = trade_prediction(input_data)
        st.success(prediction_result)

    st.success(Predict)
    st.image('path_to_your_image.jpg', caption='Your Image Caption', use_column_width=True)
    st.markdown('Contact: [LinkedIn](https://www.linkedin.com/in/ashishgoyal1018)')
if __name__== '__main__':
    main()
