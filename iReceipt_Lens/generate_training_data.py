import pickle
import pandas as pd
import os
from receipt_data_extract import training_data_tesseract
from receipt_data_extract import training_data_tectract

train_data_tes = training_data_tesseract(image_file_dir = "receipt_data/input_images/", text_file_dir = "receipt_data/input_text/")
train_data_tex = training_data_textract(image_file_dir = "receipt_data/input_images/", text_file_dir = "receipt_data/input_text/")
with open("train_tesseract.pickle",'wb') as f:
	pickle.dump(train_data_tes,f)

data = pd.DataFrame(train_data_tex)
data.to_csv('train_textract.csv',index=False)
