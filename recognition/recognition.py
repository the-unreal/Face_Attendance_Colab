from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import tensorflow
import argparse
from pathlib import Path
from sklearn.externals import joblib

"""from embedding import get_embedding
from detect import extract_face"""

def fetch_roll(img, model_sk, emb):
  out_encoder = joblib.load('/content/Face_Attendance_Colab/recognition/model/encoder_model.pkl')
  samples = expand_dims(emb, axis=0)
  yhat_class = model_sk.predict(samples)
  predict_names = out_encoder.inverse_transform(yhat_class)
  return predict_names[0]

def recognise(image, model):
  data = load('files\\8th_sem_data.npz')
  testX_faces = data['arr_2']
  data = load('files\\8th_sem-faces-embeddings.npz')
  trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    
  in_encoder = Normalizer(norm='l2')
  trainX = in_encoder.transform(trainX)
  testX = in_encoder.transform(testX)
  out_encoder = LabelEncoder()
  out_encoder.fit(trainy)
  trainy = out_encoder.transform(trainy)
  testy = out_encoder.transform(testy)
  model_sk = SVC(kernel='linear', probability=True)
  model_sk.fit(trainX, trainy)

  img = extract_face(image)
  face_emb = get_embedding(model, img)
  fetch_roll(img, model_sk, face_emb)
    
if __name__ == "__main__":
  model = tensorflow.keras.models.load_model('model\\facenet_keras.h5')
  parser = argparse.ArgumentParser()
  parser.add_argument("file_path", type=Path)
  p = parser.parse_args()
  image = p.file_path
  recognise(image, model)