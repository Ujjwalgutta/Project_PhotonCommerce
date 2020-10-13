import os
import spacy



def save_model(output_dir):
	nlp = spacy.load(output_dir)
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)