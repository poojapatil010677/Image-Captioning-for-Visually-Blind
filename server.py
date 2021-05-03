from flask import Flask, request, render_template,send_file
import os
from PIL import Image
import numpy
from flask_ngrok import run_with_ngrok
import pytesseract
import os
try:
 from PIL import Image
except ImportError:
 import Image
from prediction import extract_features, generate_desc
from keras.models import load_model

 def getCaption(imgname):
	#img = Image.open("photo.jpg")
	img = Image.open(imgname)
	pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract"
	result = pytesseract.image_to_string(img)  
	return result

def getCaptionObject(filename):
    # load the tokenizer
    tokenizer = load(open('/content/drive/MyDrive/Capstone Project/Dataset - NEW/tokenizer.pkl', 'rb'))
    # pre-define the max sequence length (from training)
    max_length = 31
    # load the model
    model = load_model('/content/drive/MyDrive/Capstone Project/Dataset - NEW/model-ep003-loss3.335-val_loss3.561.h5')
    # load and prepare the photograph
    photo = extract_features(filename)
    #photo = extract_features("/content/drive/MyDrive/Capstone Project/Dataset - NEW/capstoneIC_dataset/"+filename)
    # generate description
    description = generate_desc(model, tokenizer, photo, max_length)
    l = description.split(' ')
    description = ' '.join(l[1:len(l)-1])
    print(description)
    return description

app = Flask(__name__)
run_with_ngrok(app) 

@app.route('/', methods=['GET', 'POST'])
def test_request():
	return "connected..."


@app.route('/caption', methods=['GET', 'POST'])
def handle_request():
	file = request.files['file']
	print(file.filename)
	file.save(file.filename)
	ans = getCaptionObject(file.filename)
	return ans

@app.route('/ocr', methods=['GET', 'POST'])
def handle_request2():
	file = request.files['file']
	print(file.filename)
	file.save(file.filename)

	#audio = "cap.mp3"

	ans = getCaption(file.filename)
	print(ans)
	return ans


if __name__ == '__main__':
	app.run()