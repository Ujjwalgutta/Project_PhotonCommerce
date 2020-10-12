# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 10:51:14 2020

@author: ujjwa
"""
import pickle
import boto3
import os
import pytesseract
import pandas as pd
import numpy as np
import re
import cv2
#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


#Get the Total company names for tesseract performance Analysis
def get_company_names_tesseract(image_file_dir = "receipt_data/input_images/", text_file_dir = "receipt_data/input_text/"):
    train_data_company = []
    image_list = os.listdir(image_file_dir)
    file_list = os.listdir(text_file_dir)
    for i in range(len(image_list)):
        with open(text_file_dir + file_list[i]) as file_data:
            my_dict_company ={}
            lines = file_data.readlines()
            if(len(lines)<6):
                continue
            del lines[0]
            del lines[4]
            del lines[2]
            del lines[1]
            del lines[1]
            line = lines[0]
            val = line.split(':')[-1]
            val1 = val[2:len(val)-2]
            val2 = re.sub(r'[\"]','',val1)
            my_dict_company['company'] = val2
        raw_text = []
        image = cv2.imread(image_file_dir + image_list[i])
        image_resize = cv2.resize(image,(400,800), interpolation = cv2.INTER_AREA)
        grayImage = cv2.cvtColor(image_resize, cv2.COLOR_BGR2GRAY)
        grayImage1 = grayImage[1:250]
        
        raw_text = pytesseract.image_to_string(grayImage1, config = '--psm 6')
        matches = []
        match = re.search(my_dict_company['company'],raw_text)
        if match:
            start = match.start()
            end = match.end()
            matches.append((start,end,'company'))
        train_data_company.append((raw_text,{"entities":matches}))
        
    cnter = 0
    company_names = []
    for j in range(len(train_data_company)):
        x,y = train_data_company[j]
        if y['entities']:
            company_names.append(j)
            cnter += 1
    return train_data_company,cnter
    


#Get the training data for Spacy using Tesseract
def training_data_tesseract(image_file_dir = "receipt_data/input_images/", text_file_dir = "receipt_data/input_text/"):
    print("This is a Program to preprocess the raw text and load the Tesseract training data \n")
	train_data_resized = []
    image_list = os.listdir(image_file_dir)
    file_list = os.listdir(text_file_dir)
    keys = ['company','date','total']
    for i in range(len(image_list)):
        with open(text_file_dir + file_list[i]) as file_data:
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
        raw_text = []
        image = cv2.imread(image_file_dir + image_list[i])
        image_resize = cv2.resize(image,(400,800), interpolation = cv2.INTER_AREA)
        grayImage = cv2.cvtColor(image_resize, cv2.COLOR_BGR2GRAY)
        
        raw_text = pytesseract.image_to_string(grayImage, config = '--psm 6')
        matches = []
        for i in range(len(keys)):
            match = re.search(my_dict[keys[i]],raw_text)
            if match:
                start = match.start()
                end = match.end()
                matches.append((start,end,str(keys[i])))
        train_data_resized.append((raw_text,{"entities":matches}))
    return train_data_resized

#Get the training data for Spacy using Textract
def training_data_textract(image_file_dir = "receipt_data/input_images/", text_file_dir = "receipt_data/input_text/"):
	print("This is a Program to preprocess the raw text and load the Textract training data \n")
	train_data_textract = []
	image_list = os.listdir(image_file_dir)
    file_list = os.listdir(text_file_dir)
	keys = ['company','date','address','total']
	for i in range(len(image_list)):
        with open(text_file_dir + file_list[i]) as file_data:
            my_dict ={}
			lines = file_data.readlines()
			if(len(lines)<6):
				continue
			del lines[0]
			del lines[4]
			j = 0
			for line in lines:
				val = line.split(':')[-1]
				val1 = val[2:len(val)-2]
				val2 = re.sub(r'[\"]','',val1)
				my_dict[keys[j]] = val2
				j += 1
		raw_text = []
		image = file_list[i][:-4]
		image_name = image + ".jpg"
		with open(image_file_dir + image_name, 'rb') as document:
			imageBytes = bytearray(document.read())
			document.close()
		print("File Processed:",i)
		textract = boto3.client('textract')
		response = textract.detect_document_text(Document={'Bytes': imageBytes})
		raw_text = ""
		for item in response["Blocks"]:
			if item["BlockType"] == "LINE":
				raw_text = raw_text + " " + item["Text"]
		matches = []
		for i in range(len(keys)):
			match = re.search(my_dict[keys[i]],raw_text)
			if match:
				start = match.start()
				end = match.end()
				matches.append((start,end,str(keys[i])))
				print("\nMatches are",matches)
		train_data_textract.append((raw_text,{"entities":matches}))
	return train_data_textract
