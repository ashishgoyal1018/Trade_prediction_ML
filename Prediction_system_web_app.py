# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import streamlit as st

import pickle

import os
os.getcwd()

#os.chdir('C:\\Users\\Lenovo\\Downloads')
#loaded_model = pickle.load('trained_model.sav','rb')

with open('trained_model.sav','rb') as file:
        loaded_model = pickle.load(file)

def trade_prediction(input_data):
# changing the input_data to numpy array
   input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
   input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

   prediction = loaded_model.predict(input_data_reshaped)
   print(prediction)

   if (prediction[0] == 0):
        return'The trade has more than 85% chances of hitting the target'
   else:
        return'The trade has more than 85% chances of hitting the stoploss'
  
  
def main():
    st.title('Day Trade Prediction ML Model')
    
    Date= st.number_input('Enter date of trade w/o month/year',min_value=1,max_value=31,value=15)
    dropdown_options_1= ["Mon", "Tue", "Wed","Thur","Fri"]
    Day= st.selectbox("Select the day:", dropdown_options_1)
    dropdown_options_2= ["Banknifty", "Nifty"]
    Symbol= st.selectbox("Select the Symbol:", dropdown_options_2)
    dropdown_options_3= ["3min", "5min","10min","15min"]
    Time_Frame= st.selectbox("Select the timeframe:", dropdown_options_3)
    dropdown_options_4= ["Morning", "Afternoon"]
    Time_of_BC_candle= st.selectbox("Select the timing:", dropdown_options_4)
    dropdown_options_5= ["0", "1"]
    Distribution_0_yes_1_no= st.selectbox("Whether chart pattern as per image, if yes select 0, else 1", dropdown_options_5)
    India_Vix= st.number_input('Enter the india vix at time of trade',min_value=13,max_value=26,value=19)
    dropdown_options_6= ["0", "1"]
    Trend_1hr_Chart_1_Bullish_0_Bearish= st.selectbox("1hr trend is bullish select 1, bearish 0", dropdown_options_6)
    dropdown_options_7= ["0", "1"]
    Trend_Day_Chart_1_Bullish_0_Bearish= st.selectbox("Day trend is bullish select 1, bearish 0", dropdown_options_7)
    dropdown_options_8= ["0", "1"]
    Gap_up_0_down_1= st.selectbox("If Gap up Select 0, for gap down 1", dropdown_options_8)
    dropdown_options_9= ["0", "1"]
    Twice_high_volume_0_Yes_1_No= st.selectbox("BC volume double the bearish candle volume select 0, else 1", dropdown_options_9)

    Predict = ''

    if st.button('Trade prediction result'):
        Predict = trade_prediction([Date,Day,Symbol,Time_Frame,Time_of_BC_candle,Distribution_0_yes_1_no,India_Vix,Trend_1hr_Chart_1_Bullish_0_Bearish,Trend_Day_Chart_1_Bullish_0_Bearish,Gap_up_0_down_1,Twice_high_volume_0_Yes_1_No])

    st.success(Predict)

if __name__== '__main__':
    main()
