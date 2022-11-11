
import pickle
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import base64
import plotly.graph_objects as go
import matplotlib.pyplot as plt


st.set_page_config(page_title="Credit Card Fraud Detection App")
import zipfile
import pandas as pd

zf = zipfile.ZipFile('Final_data.zip') 
# if you want to see all files inside zip folder
print(zf.namelist())

# now read your csv file 
df = pd.read_csv(zf.open('Final_data.csv'), encoding="utf-8")

# unzipped_file = zipfile.ZipFile("Final_data.zip", "r") # get contents without extracting.
# df = unzipped_file.read("Final_data.csv")

# Import the dataset
# fraud = pd.read_csv("https://github.com/zubeydeulukok/Fraud_Detection/edit/main/Final_data.csv")
# df = fraud.copy()
csv = df.to_csv(index=False)
b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions

model = pickle.load(open("final_model_fraud_detection", "rb"))

#separating X and y
X = df.drop("Class", axis=1)
y = df["Class"]

#Adding background image from your-local
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('./rm222batch5-kul-03.jpg') 

# # Adding background image from url
# # def add_bg_from_url():
# #     st.markdown(
# #          f"""
# #          <style>
# #          .stApp {{
# #              background-image: url("https://img.freepik.com/free-vector/business-doodle-vector-human-resources-concept_53876-126582.jpg?w=996&t=st=1667582934~exp=1667583534~hmac=d65d1b36f1eb5cb85128167529e735d1e39c9be971a0cc556aa4ce4f339e9df5");
# #              background-attachment: fixed;
# #              background-size: cover
# #          }}
# #          </style>
# #          """,
# #          unsafe_allow_html=True
# #      )

# # add_bg_from_url() 

st.markdown("<h1 style='text-align: center;border: solid; color: black;'>Credit Card Fraud Detection App</h1>", unsafe_allow_html=True)
st.write("""
This app is created to predict **Fraud Detection**. 

     """)
#add image
img = Image.open("./title_image_1.png")
col1, col2, col3 = st.columns([1,8,1]) 
with col2:
    st.image(img,caption="Credit Card Fraud",width = 500)

#To download the dataset    
href = f'<a href="data:file/csv;base64,{b64}" download="churn_data.csv">Download CSV File</a>'
st.markdown(href, unsafe_allow_html=True)

#Subheader
st.subheader('User Input Features')



def sidebar_bg(side_bg):

   side_bg_ext = 'png'

   st.markdown(
      f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
          background: cover;
      }}
      </style>
      """,
      unsafe_allow_html=True,
      )
    

# inserting image at sidebar    
# st.sidebar.image("./card.jpg", use_column_width=True)

# Sidebar header
# sidebar_bg("./sidebar.jpg")

result = '''<p style="color:#FFFFFF; border-color:#8dc6ff; font-size: 18px; background-color:#4d80f0;  border-radius: 12px; text-align: center; background-size: 200px 150px;">
<b>Upload your CSV file</b> </p>'''
st.sidebar.markdown(result, unsafe_allow_html=True)

#To show uploaded data or original dataset
uploaded_file = st.sidebar.file_uploader("", type=["csv"])
st.sidebar.markdown("")
st.sidebar.markdown("")

if uploaded_file is not None:
   st.table(df)
else:
   st.markdown("<p5 style='text-align: left;color: #FFFFFF; font-size: 18px; background-color: #0c70f2'>\
   Awaiting CSV file to be uploaded or filters on the sidebar to be selected. \
   Currently using example input parameters (Please tick the checkbox to see).</p>", unsafe_allow_html=True)

#To show data   
cbox = st.checkbox("Show Data")
input_features = '''<p style="color:#FFFFFF; border-color:#8dc6ff; font-size: 18px; background-color:#4d80f0;  border-radius: 12px; text-align: center; background-size: 200px 150px;">
<b>User Input Features</b> </p>'''
st.sidebar.markdown(input_features, unsafe_allow_html=True)
st.sidebar.markdown("")

if cbox:
    st.table(df.sample(5))
# st.sidebar.header(input_features)    
#Create features on the sidebar
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():       
                
        V3 = st.sidebar.slider('V3', float(df["V3"].min()), float(df["V3"].max()), float(-1.61), 0.01)
        V4 = st.sidebar.slider('V4', float(df["V4"].min()), float(df["V4"].max()), float(4.00), 0.01)
        V7 = st.sidebar.slider('V7', float(df["V7"].min()), float(df["V7"].max()), float(-2.54), 0.01)
        V9 = st.sidebar.slider('V9', float(df["V9"].min()), float(df["V9"].max()), float(-2.77), 0.01)
        V10 = st.sidebar.slider('V10', float(df["V10"].min()), float(df["V10"].max()), float(-2.77), 0.01)
        V11 = st.sidebar.slider('V11', float(df["V11"].min()), float(df["V11"].max()), float(3.20), 0.01)
        V12 = st.sidebar.slider('V12', float(df["V12"].min()), float(df["V12"].max()), float(-2.90), 0.01)
        V14 = st.sidebar.slider('V14', float(df["V14"].min()), float(df["V14"].max()), float(-4.29), 0.01)
        V16 = st.sidebar.slider('V16', float(df["V16"].min()), float(df["V16"].max()), float(-1.14), 0.01)
        V17 = st.sidebar.slider('V17', float(df["V17"].min()), float(df["V17"].max()), float(-2.83), 0.01)
        
        data= {'V3' : V3,
                'V4' : V4,
                'V7' : V7,
                'V9' : V9,
                'V10' : V10,
                'V11' : V11,
                'V12' : V12,
                'V14' : V14,
                'V16' : V16,
                'V17' : V17   
                }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

#To see the selected filters on the main page
col1, col2, col3, col4,col5 = st.columns(5)
col1.metric("*V3", input_df["V3"].round(3)) 
col2.metric("*V4", input_df["V4"].round(3))
col3.metric("*V7", input_df["V7"].round(3))
col4.metric("*V9", input_df["V9"].round(3))
col5.metric("*V10", input_df["V10"].round(3))
col1, col2, col3,col4,col5 = st.columns(5)
col1.metric("*V11", input_df["V11"].round(3))
col2.metric("*V12", input_df["V12"].round(3)) 
col3.metric("*V14", input_df["V14"].round(3))
col4.metric("*V16", input_df["V16"].round(3))
col5.metric("*V17", input_df["V17"].round(3))
st.markdown("---")


#Check button and results on the sidebar
st.sidebar.write("Press **check** if configuration is complete.")
sample = input_df
if st.sidebar.button("Check"):
    prediction = model.predict(sample)
    prediction_proba = model.predict_proba(sample)
    if prediction == 0 :
        st.subheader("Prediction")
        result = f'<p style="color:black; border-color:#8dc6ff; font-size: 24px; background-color:#b5e7a0">\
        The transaction according to your inputs is <b>Non-Fraud</b> with the {prediction_proba[:,0][0]*100 : .1f}% probability.</p>'
        st.markdown(result, unsafe_allow_html=True)
                        
    else:
        st.subheader("Prediction")
        result = f'<p style="color:black; border-color:#8dc6ff; font-size: 24px; background-color:#f7786b">\
        The transaction according to your inputs is <b>Fraud</b> with the {prediction_proba[:,1][0]*100 : .1f}% probability.</p>'
        st.markdown(result, unsafe_allow_html=True)
    fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = (prediction_proba[:,1][0])*100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Fraud Risk (%)",'font': {'size': 24}},
            gauge = {'axis': {'range': [None, 100]},
                    'bar' : {'color':'red'}, 
                    'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50}
                    }))      
    st.plotly_chart(fig, use_container_width=True)
    
st.markdown('**Created by G-7 using Streamlit**')
