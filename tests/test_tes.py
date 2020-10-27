import spacy
import cv2
import pytesseract
from iReceipt_Lens.predict import predict_tesseract

tesseract_model_dir_best = '../models/tesseract/epoch_67'
file_name = '../test_imgs/X51005763964.jpg'
entities_known = ["YONG CEN ENTERPRISE","06/01/2018","65.70"]

def test_ocr_tesseract():
  entities = predict_tesseract(file_name, model_dir = tesseract_model_dir_best)
  assert entities["company"] == entities_known[0]
  assert entities["date"] == entities_known[1]
  assert entities["total"] == entities_known[2]
  


