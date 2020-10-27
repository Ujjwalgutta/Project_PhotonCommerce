import streamlit as st
import os
import spacy
import numpy as np
import pandas as pd
import cv2
from iReceipt_Lens.predict import predict_tesseract
from iReceipt_Lens.predict import predict_textract
import pytesseract
import json



Title_html = """
    <style>
        .title h1{
          user-select: none;
          font-size: 72px;
          color: black;
          background: repeating-linear-gradient(-45deg, red 0%, yellow 7.14%, rgb(0,255,0) 14.28%, rgb(0,255,255) 21.4%, cyan 28.56%, blue 35.7%, magenta 42.84%, red 50%);
          background-size: 600vw 600vw;
          -webkit-text-fill-color: transparent;
          -webkit-background-clip: text;
          animation: slide 10s linear infinite forwards;
        }
        @keyframes slide {
          0%{
            background-position-x: 0%;
          }
          100%{
            background-position-x: 600vw;
          }
        }
    </style> 
    
    <div class="title">
        <h1 style='text-align: center;'>iR-Lens</h1>
    </div>
    """

st.markdown(Title_html, unsafe_allow_html=True) #Title rendering

def file_selector(folder_path='test'):
    filenames = os.listdir(folder_path)
    return filenames

fname = file_selector()
fnames = ["None"]
for i in range(len(fname)):
        fnames.append(fname[i])

df = pd.DataFrame({
  'first column': fnames
  })

file_select = st.sidebar.selectbox(
    "Select the file: ",
     df['first column'])
'File selected: ', file_select[:-4]

ocr_select = st.sidebar.selectbox(
        "Select the OCR model: ",
        ("None","Amazon Textract","Google Tesseract")
        )
'Model selected: ', ocr_select

if (ocr_select == "Google Tesseract"):
        if (file_select != "None"):
                file_name = 'test/' + file_select
                c1,c2 = st.beta_columns((2,1))
                with c2:
                        st.header("Entities Extracted")
                image = cv2.imread(file_name)
                with c1:
                        st.image(image, width = 300)
                pred = predict_tesseract(file_name)
                with c2:
                        st.json(pred)
                        if st.button('SAVE'):
                                with open('data.json', 'w') as fp:
                                        json.dump(pred, fp)


if (ocr_select == "Amazon Textract"):
        if (file_select != "None"):
                file_name = 'test/' + file_select
                c1,c2 = st.beta_columns((2,1))
                with c2:
                        st.header("Entities Extracted")
                image = cv2.imread(file_name)
                with c1:
                        st.image(image, width = 300)
                pred = predict_textract(file_name)
                with c2:
					st.json(pred)
							if st.button('SAVE'):
									with open('data.json', 'w') as fp:
											json.dump(pred, fp)
