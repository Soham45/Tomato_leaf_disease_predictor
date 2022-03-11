
from flask import Flask, render_template, request
 
import numpy as np
import os
import tensorflow as tf
import json
import math 
import datetime
import time
 
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.applications.vgg16 import VGG16
vgg16 = VGG16(include_top=False, weights='imagenet')
MODEL_PATH ="C:/Users/Soham Purkar/Desktop/BE_Project/Website_Flask/VGG16.h5"
print("MODEL LOADED")
model = load_model(MODEL_PATH)
def mapping(output):
  for index,value in enumerate(output):
    if value > 0.65:
      return index
def read_image(file_path):
    print("[INFO] loading and preprocessing image…") 
    image = load_img(file_path, target_size=(224, 224)) 
    image = img_to_array(image) 
    image = np.expand_dims(image, axis=0)
    image /= 255. 
    return image
def test_single_image(path):
  diseases = ['Bacterial_spot','Early_blight','Late_blight','Leaf_Mold','Septoria_leaf_spot','Spider_mites Two-spotted_spider_mite','Target_Spot','Tomato_Yellow_Leaf_Curl_Virus','Tomato_mosaic_virus','Tomato___healthy']
  images = read_image(path)
  time.sleep(.5)
  bt_prediction = vgg16.predict(images) 
  res = model.predict(bt_prediction)
  
  for idx, diseases, x in zip(range(0,11), diseases , res[0]):
   print("ID: {}, Label: {} {}%".format(idx, diseases, round(x*100,2) ))
  
  for x in range(3):
   print('.'*(x+1))
   time.sleep(.2)
  class_predicted = model.predict(bt_prediction)
  class_dictionary = {'Tomato___Bacterial_spot': 0, 'Tomato___Early_blight': 1, 'Tomato___Late_blight': 2, 'Tomato___Leaf_Mold': 3, 'Tomato___Septoria_leaf_spot': 4, 'Tomato___Spider_mites Two-spotted_spider_mite': 5, 'Tomato___Target_Spot': 6, 'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 7, 'Tomato___Tomato_mosaic_virus': 8, 'Tomato___healthy': 9}
  inv_map = {v: k for k, v in class_dictionary.items()}
  z=mapping(class_predicted[0])

  print("----------------------------------------------------------------------------------------")
  print("\t\t\t\t\tFinal Output")
  print("----------------------------------------------------------------------------------------")
  label=inv_map[z];
  print("ID: {}, Label: {}".format(z,  inv_map[z]))      
    
  return label

app = Flask(__name__)
@app.route("/", methods=['GET'])
def index():
    return render_template('index.html')


@app.route("/predict", methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, './uploads', f.filename)
        f.save(file_path)
        preds = test_single_image(file_path)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(port=5001,debug=True)