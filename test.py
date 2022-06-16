#importing libraries
import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers.core import Dense
from keras.models import model_from_json
import os
import cv2

#loading tha model
json_file=open('modell.json','r')
loaded_model_json=json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)


print(loaded_model)
#load weights into new model
pathformodell="E:\AI and ML-pr\day14\modell.h5"
try:
    new_model=loaded_model.load_weights(pathformodell,by_name = False, skip_mismatch = False, options = None)
    print("Model loaded succesfully***")
except:
    print("Model Not loaded!!!")

print(new_model)

label=['Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
       'Blueberry___healthy','Cherry_(including_sour)___healthy','Cherry_(including_sour)___Powdery_mildew',
       'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_',
       'Corn_(maize)___healthy','Corn_(maize)___Northern_Leaf_Blight','Grape___Black_rot','Grape___Esca_(Black_Measles)',
       'Grape___healthy','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot',
       'Peach___healthy','Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight',
       'Potato___healthy','Potato___Late_blight','Raspberry___healthy','Soybean___healthy',
       'Squash___Powdery_mildew','Strawberry___healthy','Strawberry___Leaf_scorch','Tomato___Bacterial_spot',
       'Tomato___Early_blight','Tomato___healthy','Tomato___Late_blight','Tomato___Leaf_Mold',
       'Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot',
       'Tomato___Tomato_mosaic_virus','Tomato___Tomato_Yellow_Leaf_Curl_Virus']

#path=input("Enter your image path-: ")
test_image=image.load_img(path,target_size=(128,128))
#print(test_image)
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result = loaded_model.predict(test_image)

#print(f"Result is --> {result}")
fresult=np.max(result)
label2=label[result.argmax()]
print(f"your leaf disease is --> {label2}")
