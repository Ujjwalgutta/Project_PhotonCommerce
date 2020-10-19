# Script to train the Spacy Model using OCR extracted data
import os
import time
import boto3
from sklearn.model_selection import train_test_split
import pickle
import cv2
import pytesseract
import spacy
#Comment out the next line if you are using a CPU to train your model
spacy.prefer_gpu()
from utils import evaluate_model()
from utils import save_model()
import random
from spacy.util import minibatch, compounding
from pathlib import Path
from spacy.gold import GoldParse
from spacy.scorer import Scorer
import pandas as pd

textract_model_dir_best = 'models/textract/model_blank'
tesseract_model_dir_best = 'models/tesseract/model_blank'

# Train new NER model
def train_new_NER(model=None, output_dir, n_iter=100):
    #Load the model, set up the pipeline and train the entity recognizer.
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    # create the built-in pipeline components and add them to the pipeline
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")

    # add labels
    for _, annotations in training_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        # reset and initialize the weights randomly – but only if we're training a new model
        if model is None:
            nlp.begin_training()
            print("Training Started...")
        history_blank = []
        for itn in range(n_iter):
            random.shuffle(training_data)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(training_data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.2,  # dropout - make it harder to memorize data
                    losses=losses,
                )
            print("Losses", losses)
            epoch_path = output_dir + 'epoch_' + str(itn)
            nlp.to_disk(epoch_path)
            if val is not None:
                score_prf = evaluate_model(nlp,val)
            history_blank.append({"epoch": itn, "losses": losses, "Precision": score_prf['ents_p'], "Recall": score_prf['ents_r'], "F1-score": score_prf['ents_f']})
    
	data = pd.DataFrame(history_blank)
	data.to_csv('history_blank_model.csv',index=False)
	return nlp

# Inference
def predict_textract(file_name, model_dir = textract_best_model_dir):
    documentName = ""
    documentName = 'test_imgs/' + file_name
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
  

# Main loop to call all the functions
if __name__ == "__main__":
    '''
    # To use Tesseract extracted training data
    with open('data/processed/train_tesseract.pickle','rb') as f:
        train_data_tes = pickle.load(f)
    training_data,testing_data = train_test_split(train_data_tes,test_size = 0.15, random_state = 42)
    test,val = train_test_split(testing_data, test_size = 0.5, random_state = 42)
    tes_nlp = train_new_model(output_dir = tesseract_model_dir_best)
    '''
    
    # To use Textract extracted training data
    data = pd.read_csv('data/processed/train_textract.csv')
    data_list = data.values.tolist()
    train_data_tex = [tuple(l) for l in data_list]
    for k in range(len(train_data_tex)):
        x,y = train_data_tex[k]
        y = eval(y)
        train_data_tex[k] = x,y
    training_data,testing_data = train_test_split(train_data_tex,test_size = 0.15, random_state = 42)
    test,val = train_test_split(testing_data, test_size = 0.5, random_state = 42)
    tex_nlp = train_new_model(output_dir = textract_model_dir_best)
    
