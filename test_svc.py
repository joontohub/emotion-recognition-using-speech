from emotion_recognition import EmotionRecognizer
from sklearn.svm import SVC
import os
# init a model, let's use SVC
my_model = SVC()
# pass my model to EmotionRecognizer instance
# and balance the dataset
rec = EmotionRecognizer(model=my_model, emotions=['sad', 'neutral', 'happy'], balance=True, verbose=0)
# train the model
rec.train()
# check the test accuracy for that model
print("Test score:", rec.test_score())
# check the train accuracy for that model
print("Train score:", rec.train_score())

# loads the best estimators from `grid` folder that was searched by GridSearchCV in `grid_search.py`,
# and set the model to the best in terms of test score, and then train it

# this is a neutral speech from emo-db
#print("Prediction:", rec.predict("data/emodb/wav/15a04Nc.wav"))
# this is a sad speech from TESS

files = os.listdir("./KsponSpeech_0001_wav/")

for i in files :


    predict_filename = "./KsponSpeech_0001_wav/" + i
    prediction = rec.predict(predict_filename)


    print(i)
    
    print("Prediction:", rec.predict("./KsponSpeech_0001_wav/./KsponSpeech_000001.wav"))

    
    print('-----------------------------------')
    #테스트를 위해 추가
    

