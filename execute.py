import tensorflow
from sklearn.externals import joblib

from face_recognition_attendance.recognition import *
from face_recognition_attendance.embedding import *
from face_recognition_attendance.detect import *

def recog(img, model):
  model_sk = joblib.load('/content/face_recognition_attendance/model/8th_sem_faces.pkl')
  image = extract_face(img)
  face_emb = get_embedding(model, image)
  fetch_roll(img, model_sk, face_emb)

if __name__ == "__main__":
  model = tensorflow.keras.models.load_model('/content/facenet_keras.h5')
  img = "/content/frame72.jpg"
  recog(img, model)