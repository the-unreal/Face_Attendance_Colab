import tensorflow
from sklearn.externals import joblib
import cv2
import argparse
from pathlib import Path

from recognition.recognition import *
from recognition.embedding import *
from recognition.detect import *

from liveness.detection import *

def spoof(img, model):
  image = cv2.imread(img)
  faceCascade = cv2.CascadeClassifier('/content/Face_Attendance_Colab/liveness/haarcascade_frontalface_default.xml')
  return predictperson(image, model, faceCascade)

def recog(img, model):
  model_sk = joblib.load('/content/Face_Attendance_Colab/recognition/model/8th_sem_faces.pkl')
  image = extract_face(img)
  face_emb = get_embedding(model, image)
  return fetch_roll(img, model_sk, face_emb)

if __name__ == "__main__":
  model_recog = tensorflow.keras.models.load_model('/content/Face_Attendance_Colab/recognition/model/facenet_keras.h5')
  model_spoof = tensorflow.keras.models.load_model('/content/Face_Attendance_Colab/liveness/models/spoof.hdf5')
  parser = argparse.ArgumentParser()
  parser.add_argument("file_path")
  p = parser.parse_args()
  img = p.file_path
  args = vars(parser.parse_args())
  spoof_result = spoof(args["file_path"], model_spoof)
  if spoof_result == 'real':
    print(recog(img, model_recog))
  else:
    print(spoof_result)