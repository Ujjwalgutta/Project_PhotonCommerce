# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 11:52:51 2020

@author: ujjwa
"""
import boto3
import cv2
import pytesseract
import spacy


tesseract_model_dir_best = '../models/tesseract_model_blank/epoch_67'
textract_model_dir_best = '../models/textract_model_blank/epoch_65'


def predict_tesseract(file_name, model_dir = tesseract_model_dir_best):
    image = cv2.imread(file_name)
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raw_text = pytesseract.image_to_string(grayImage, config = '--psm 6')
    nlp = spacy.load(updated_model_dir)
    doc = nlp(raw_text)
    pred = {}
    for ent in doc.ents:
        if ent.label_ in ["company","date","total"]:
            pred[ent.label_] = ent.text
    return pred

def predict_textract(file_name, model_dir = textract_best_model_dir):
    documentName = ""
    documentName = 'test/' + file_name
    with open(documentName, 'rb') as document:
        imageBytes  = bytearray(document.read())
    textract = boto3.client('textract')
    response = textract.detect_document_text(Document={'Bytes': imageBytes})
    raw_text = ""
    for item in response["Blocks"]:
        if item["BlockType"] == "LINE":
            raw_text = raw_text + " " + item["Text"]
    
    nlp = spacy.load(model_dir)
    doc = nlp(raw_text)
    pred = {}
    for ent in doc.ents:
        if ent.label_ in ["company","date","total"]:
            pred[ent.label_] = ent.text 
    return pred

