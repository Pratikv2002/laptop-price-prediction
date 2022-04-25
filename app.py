import streamlit as st
import pickle
import numpy as np

# import the model
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title("Laptop Price Predictor")

# brand
company = st.selectbox('Brand',df['Company'].unique())

# type of laptop
type = st.selectbox('Type',df['TypeName'].unique())

# Ram
ram = st.selectbox('Ram(in GB)',[2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('Weight')

# Touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

# ips
ips = st.selectbox('IPS',['No','Yes'])

# screen size

screen_size = st.number_input('Screen Size')

resolution = st.selectbox('Screen Resolution',['1920X1080','1366X768','1600X900','3840X2160','3200X1800','2880X1800','2560X1600','2560X1440','2304X1440'])

# cpu
cpu = st.selectbox('CPU',df['Cpu brand'].unique())

# hdd
hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

# ssd
ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

# gpu
gpu = st.selectbox('GPU',df['Gpu brand'].unique())

# os
os = st.selectbox('OS',df['os'].unique())
if st.button('Predict Price'):
    ppi = None
    if touchscreen == 'Yes':
        touchscreen=1
    else:
        touchscreen=0

    if ips == 'Yes':
        ips=1
    else:
        ips=0

    X_res = int(resolution.split('X')[0])
    Y_res = int(resolution.split('X')[1])

    ppi = ((X_res**2)+(Y_res**2))**0.5/screen_size
    query = np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])
    query=query.reshape(1,12)
    st.title("The predictec price of the configuration is:"+str(int(np.exp(pipe.predict(query)))))
