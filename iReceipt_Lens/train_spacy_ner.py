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

# Train new NER model
def train_new_NER(model=None, output_dir=textract_model_dir, n_iter=100):
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
            epoch_path = 'model_blank/epoch_' + str(itn)
            nlp.to_disk(epoch_path)
            if val is not None:
                score_prf = evaluate_model(nlp,val)
            history_blank.append({"epoch": itn, "losses": losses, "Precision": score_prf['ents_p'], "Recall": score_prf['ents_r'], "F1-score": score_prf['ents_f']})
    
	data = pd.DataFrame(history_blank)
	data.to_csv('history_blank_model.csv',index=False)
	return nlp

## Update existing spacy model and store into a folder
def update_model(model='en_core_web_sm', output_dir=updated_model_dir_large, n_iter=100):
   #Load the model, set up the pipeline and train the entity recognizer.
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
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
        # reset and initialize the weights randomly – but only if we're
        # training a new model
        print("Training model...")
        final_loss = []
        if model is None:
            nlp.begin_training()
        else:
            optimizer = nlp.resume_training()
        history_pretrained = []
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
                    drop=0.2,  # dropout - make it harder to memorise data
                    sgd = optimizer,
                    losses=losses,
                )
			
			print("Losses", losses)
            epoch_path = 'model_pretrained/epoch_' + str(itn)
            nlp.to_disk(epoch_path)  # Make sure you don't use the SpaCy's large model because each model occupies 786 MB of data.
            if val is not None:
                score_prf = evaluate_model(nlp,val)
            history_pretrained.append({"Epoch": itn, "losses": losses, "Precision": score_prf['ents_p'], "Recall": score_prf['ents_r'], "F1-score": score_prf['ents_f']})
	
	data = pd.DataFrame(history_pretrained)
	data.to_csv('history_pretrained_model.csv',index=False)
	save_model(nlp, output_dir)
	return nlp
	
