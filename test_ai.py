import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from deep_emotion_recognition import DeepEmotionRecognizer
import asyncio

pathFile = './UnityVoices/voice_recording.wav'

switch = False

count = 0
def PredictUnityEmotion():
    deeprec = DeepEmotionRecognizer(emotions=['angry', 'sad', 'neutral', 'ps', 'happy'], n_rnn_layers=2, n_dense_layers=2, rnn_units=128, dense_units=128)
    # train the model
    deeprec.train()
    # get the accuracy
    print(deeprec.test_score())
    # predict angry audio sample
    prediction = deeprec.predict("./UnityVoices/voice_recording.wav")
    print(f"Prediction: {prediction}")



while True:
    if (os.path.isfile(pathFile) and count == 0):
        switch = True
    elif count == 1:
        print("already activated")
    else:
        print("wait")

    if (switch):
        PredictUnityEmotion()
        print("----------------------result------------------- ")
        switch = False
        count += 1
    else:
        print("doesnt activate")
    time.sleep(1)
  


