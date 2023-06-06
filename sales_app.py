import streamlit as st
import pickle
import sklearn
import pandas as pd
import numpy as np
from PIL import Image

model1 = pickle.load(open('model1.sav', 'rb'))
model2 = pickle.load(open('model2.sav', 'rb'))
model3 = pickle.load(open('model3.sav', 'rb'))

st.title('Advertising Sales Prediction')
st.sidebar.header('Advertising Cost Data')
image= Image.open('FSP1.jpeg')

st.image(image,'')


def sales_report():
    Tv=st.sidebar.slider('Tv',100,200,10)
    Radio=st.sidebar.slider('Radio',1,50,1)
    Newspaper=st.sidebar.slider('Newspaper',1,120,1)

    sales_report_data = {
        'Tv':Tv,
        'Radio':Radio,
        'Newspaper':Newspaper
    }
    sales_data=pd.DataFrame(sales_report_data,index=[0])
    return sales_data

entered_data=sales_report()
st.header('Sales Data')
# Display the table
st.write(entered_data)

models = {
    'Linear Regression': model1,
    'Decison Tree': model2,
    'Random Forest': model3
}
# Create a dropdown menu to select the model
selected = st.sidebar.selectbox('Select a model', list(models.keys()))
selected_model=models[selected]
sales=selected_model.predict(entered_data)
st.subheader('Sales Done')
st.subheader(str(np.round(sales[0])))
