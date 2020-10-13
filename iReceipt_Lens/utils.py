import os
import spacy
from spacy.util import minibatch, compounding
from pathlib import Path
from spacy.gold import GoldParse
from spacy.scorer import Scorer
import pandas as pd


def save_model(model, output_dir):
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        model.to_disk(output_dir)
        print("Saved model to", output_dir)
		

def evaluate_model(model,examples):
    scorer = Scorer()
    for input_, annot in examples:
        doc_gold_text = model.make_doc(input_)
        gold = GoldParse(doc_gold_text, entities=annot['entities'])
        pred_value = model(input_)
        scorer.score(pred_value, gold)
    return scorer.scores
