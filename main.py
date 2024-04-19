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
import pandas as pd

new = pd.read_csv(r"C:\Users\User\Downloads\requirements.txt")
new.head()

with st.sidebar:
    st.title("Description:")
    st.markdown('This Model basically brings in Conversational Text data'
                ' & performs Sentiment Analysis on individual response.'
                ' We have used pre-trained Transformer Model with Streamlit App.'
                ' The spectrum has 3 scales on it for every response as below:')
    #st.title('😃 - Strongly Positive')
    st.title('😃 - Positive')
    st.title('😑 - Neutral')
    st.title('😠 - Negative')
    #st.title('😠 - Strongly Negative')
    st.title('Developed by Sayan Roy')
st.title("Sentiment Analysis")
st.markdown(" On Conversational Data ")
st.title("💻<------------------->💻")

csv = st.file_uploader("Please upload any .CSV")

if csv is not None:
    df = pd.read_csv(csv)
    st.write(df.head())
    st.write("This file has ",df.shape[0],"rows & ",df.shape[1],"columns.")
    col = st.text_input("Please enter the name of the Column which has Text data...",placeholder="Enter a Column Name...")
    if col is not '':
        if col not in df.columns:
            st.write("Column name doesnt exist")
            st.write("Please Try Again")
        else:
            df['Flag'] = df[col].apply(lambda x: x if x is not np.nan else "Missing")
            df1 = df[df['Flag'] != "Missing"]
            df2 = df1["Text"].values.tolist()
            #st.write(df2[5])
            #st.write(len(df2[5].split()))
            #st.write(type(df2))
            clf = pipeline("sentiment-analysis")
            label = []
            #new = df2[0:5]
            #label = [clf(df2[0])[0]['label']]
            #for i in range(5):
            cnt=0
            for i in range(len(df2)):
                if len(df2[i].split()) < 512:
                    label.extend([clf(df2[i])[0]['label']])
                else:
                    cnt=cnt+1
            if cnt==1:
                st.write(cnt, " row detected where token length > 512, hence discarded.")
            else:
                st.write(cnt, " rows detected where token length > 512, hence discarded.")
            #label.extend([clf(new[i])[0]['label']])
            df3 = pd.DataFrame(data=list(zip(df2,label)),columns=["Text","Label"])
            df4 = df3.groupby(by=['Label']).count()
            st.bar_chart(data=df4,use_container_width=True)
            final = df3.to_csv().encode('utf-8')
            but1 = st.download_button(
                label="Download the Result as CSV",
                data=final,
                file_name='Sentiment_Analysis.csv',
                mime='text/csv'
            )

