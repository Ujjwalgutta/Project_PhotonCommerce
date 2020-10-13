import pickle
import pandas as pd
import os
from receipt_data_extract import training_data_tesseract
from receipt_data_extract import training_data_tectract

train_data_tes = training_data_tesseract()
train_data_tex = training_data_textract()
with open("train_data/train_tesseract.pickle",'wb') as f:
	pickle.dump(train_data_tes,f)

data = pd.DataFrame(train_data_tex)
data.to_csv('train_data/train_textract.csv',index=False)
