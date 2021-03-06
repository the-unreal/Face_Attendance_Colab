from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2

def predictperson(frame, model, faceCascade):
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
  faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30))
  height, width, channels = frame.shape
        
  faces_inside_box = 0
        
  for (x, y, w, h) in faces:
    
    faces_inside_box += 1
    
    if faces_inside_box == 1:
      cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
      image = cv2.resize(frame, (128, 128))
      image = image.astype("float") / 255.0
      image = img_to_array(image)
      image = np.expand_dims(image, axis=0)
      (fake, real) = model.predict(image)[0]
      
      if fake > real:
        label = "real"
      else:
        label = "fake"
    
  return label
  
        
if __name__ == "__main__":
  model = load_model("/content/Liveness-Detection/models/the-unreal.hdf5")
  img = cv2.imread("/content/Liveness-Detection/unreal2.jpg")
  cascPath = "/content/Liveness-Detection/haarcascade_frontalface_default.xml"
  faceCascade = cv2.CascadeClassifier(cascPath)
  predictperson(img, model)