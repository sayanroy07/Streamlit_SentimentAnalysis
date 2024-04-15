#from transformers import pipeline
#import torch
#if torch.cuda.is_available():
#    print("GPU")
#else:
#    print("CPU")
#classifier = pipeline("sentiment-analysis")
#data = ["I love you", "I hate you"]
#classifier(data)

import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline

with st.sidebar:
    st.title("Description:")
    st.markdown('This Model basically brings in Conversational Text data'
                ' & performs Sentiment Analysis on individual response.'
                ' The spectrum has 5 scales on it for every response as below:')
    st.title('😃 - Strongly Positive')
    st.title('🙂 - Positive')
    st.title('😑 - Neutral')
    st.title('😔 - Negative')
    st.title('😠 - Strongly Negative')
    st.title('Developed by Sayan Roy')
st.title("Sentiment Analysis")
st.markdown(" On Conversational Data ")
st.title("💻<------------------->💻")

csv = st.file_uploader("Please upload any .CSV")
name = csv.name.split('.')[0]
if csv is not None:
    df = pd.read_csv(csv)
    st.write(df.head())
    col = st.text_input("Please enter the name of the Column which has Text data...",placeholder="Enter a Column Name...")
    if col is not '':
        if col not in df.columns:
            st.write("Column name doesnt exist")
            st.write("Please Try Again")
        else:
            df['Flag'] = df[col].apply(lambda x: x if x is not np.nan else "Missing")
            df1 = df[df['Flag'] != "Missing"]
            df2 = df1["Text"].values.tolist()
            #st.write(type(df2))
            clf = pipeline("sentiment-analysis")
            label = []
            #new = df2[0:5]
            label = [clf(df2[0])[0]['label']]
            #for i in range(5):
            for i in range(range(len(df2))):
                label.extend([clf(df2[i])[0]['label']])
                #label.extend([clf(new[i])[0]['label']])
            df3 = pd.DataFrame(data=list(zip(df2,label)),columns=["Text","Label"])
            st.write(df3)
            final = df3.to_csv().encode('utf-8')
            but1 = st.download_button(
                label="Download the Result as CSV",
                data=final,
                file_name=name+'_SA.csv',
                mime='text/csv'
            )

