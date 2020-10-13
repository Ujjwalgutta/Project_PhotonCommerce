# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 14:13:28 2020

@author: ujjwa
"""
import time
import cv2
import pickle
import os
import pytesseract
import re
import numpy as np
#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
start = time.time()
def evaluate_simple_parser(file_name = 'receipt_parser.pickle'):
    keys = ['company','date','total']
    y_true_name = []
    y_pred_name = []
    y_true_total = []
    y_pred_total = []
    y_true_date = []
    y_pred_date = []
    with open(file_name,'rb') as f:
        receipt_parser_simple = pickle.load(f)
        
    for j in range(len(receipt_parser_simple)):
        a,b = receipt_parser_simple[j]
        file_name = a[:-4] + '.txt'
        with open('receipt_data/input_text/' + file_name) as file_data:
            my_dict ={}
            lines = file_data.readlines()
            if(len(lines)<6):
                continue
            del lines[0]
            del lines[4]
            del lines[2]
            j = 0
            for line in lines:
                val = line.split(':')[-1]
                val1 = val[2:len(val)-2]
                val2 = re.sub(r'[\"]','',val1)
                my_dict[keys[j]] = val2
                j += 1
        if(b['Total Amount'] == 'Total Not Found'):
           temp = b['Total Amount']
        else:
            temp = re.sub(r'[\$\s]','',b['Total Amount'])
        y_pred_total.append(temp)
        y_pred_name.append(b['Name'])
        y_pred_date.append(b['date'])
        y_true_total.append(my_dict['total'])
        y_true_name.append(my_dict['company'])
        y_true_date.append(my_dict['date'])
    return y_true_date,y_true_name,y_true_total,y_pred_date,y_pred_total,y_pred_name
            
y_true_d,y_true_n,y_true_t,y_pred_d,y_pred_t,y_pred_n = evaluate_simple_parser()
end = time.time()
print("Time taken: ",str(end-start))


    