import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from deep_emotion_recognition import DeepEmotionRecognizer
import asyncio


##

import os, glob
 
import os.path


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

    #테스트를 위해 잠시 꺼둠
    # prediction = deeprec.predict("./UnityVoices/voice_recording.wav")

    #테스트용
   
    files = os.listdir("./KsponSpeech_0001_wav/")

    for i in files :


        predict_filename = "./KsponSpeech_0001_wav/" + i
        prediction = deeprec.predict(predict_filename)


        print(i)
        
        print(f"Prediction: {prediction}")
        print(deeprec.predict_proba(predict_filename))
        print('-----------------------------------')
        #테스트를 위해 추가
      


PredictUnityEmotion()

#테스트를 위해 잠시 꺼둠
# while True:

    
    # if (os.path.isfile(pathFile) and count == 0):
    #     switch = True
    # elif count == 1:
    #     print("already activated")
    # else:
    #     print("wait")

    # if (switch):
    #     PredictUnityEmotion()
    #     print("----------------------result------------------- ")
    #     switch = False
    #     count += 1
    # else:
    #     print("doesnt activate")
    # time.sleep(1)
  


