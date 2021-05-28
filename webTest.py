import os
import io
import sys
from flask import Flask
from flask import request , redirect, url_for, send_from_directory, jsonify, json
from werkzeug.utils import secure_filename
from PIL import Image
import base64
#import test_ai


from emotion_recognition import EmotionRecognizer

import pyaudio
import os
import time
import wave
from sys import byteorder
from array import array
from struct import pack
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
import os
from utils import get_best_estimators



# 파이썬에서는 경로 표시할때, 같은 경로이면, ./ 점을 붙혀줘야한다. 안 붙혀주면 오류남
UPLOAD_FOLDER = './UnityVoices'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(['wav'])


result_emotion = ""
result_prob = 0



#####################################

THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 16000

SILENCE = 30



pathFile = './UnityVoices/voice_recording.wav'

switch = False

count = 0

##################################



def get_estimators_name(estimators):
    result = [ '"{}"'.format(estimator.__class__.__name__) for estimator, _, _ in estimators ]
    return ','.join(result), {estimator_name.strip('"'): estimator for estimator_name, (estimator, _, _) in zip(result, estimators)}



if __name__ == "__main__":
    estimators = get_best_estimators(True)
    estimators_str, estimator_dict = get_estimators_name(estimators)
    import argparse
    parser = argparse.ArgumentParser(description="""
                                    Testing emotion recognition system using your voice, 
                                    please consider changing the model and/or parameters as you wish.
                                    """)
    parser.add_argument("-e", "--emotions", help=
                                            """Emotions to recognize separated by a comma ',', available emotions are
                                            "neutral", "calm", "happy" "sad", "angry", "fear", "disgust", "ps" (pleasant surprise)
                                            and "boredom", default is "sad,neutral,happy"
                                            """, default="sad,neutral,happy")
    parser.add_argument("-m", "--model", help=
                                        """
                                        The model to use, 8 models available are: {},
                                        default is "BaggingClassifier"
                                        """.format(estimators_str), default="BaggingClassifier")



    # Parse the arguments passed
    args = parser.parse_args()

    features = ["mfcc", "chroma", "mel"]
    detector = EmotionRecognizer(estimator_dict[args.model], emotions=args.emotions.split(","), features=features, verbose=0)
    detector.train()
    print("Test accuracy score: {:.3f}%".format(detector.test_score()*100))


##########################################################################






def allowed_file(filename):
  # this has changed from the original example because the original did not work for me
    return filename[-3:].lower() in ALLOWED_EXTENSIONS


# @app.route('/' , methods=['GET', 'POST'])
# def home():
#     if request.method == "POST":
#         data = request.files['myimage'].read()
#         imag = Image.open(io.BytesIO(data))
#         filename = 'myimage'
#         imag.save(os.path.join(app.root_path, filename))
           
#         print(type(data))
#         filename = "abc"
#         data.save(os.path.join("/simplePyweb/server/",filename))

#     return 'Hello111, World!'
@app.route('/' , methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(filename, "sdfsadf")
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("activate picture emotion detector")
            #test_ai(UPLOAD_FOLDER + "/" + filename)
            
            
            # result_emotion, result_prob = picture_detector.Detector()
            # result_prob = str(result_prob)
            # tag = "T"
            # data = tag + result_emotion + tag + result_prob
            # response = app.response_class(
            #     response=json.dumps(data),
            #     status=200,
            #     mimetype='application/json'
            # )




            prediction = detector.predict(pathFile)
            print(prediction)            
            print("----------------------result------------------- ")



            print( "json data :::: " , prediction)
            return prediction
            # for browser, add 'redirect' function on top of 'url_for'
            #return url_for("result_page", data=data)
    else:
        return "Hello world it is for post "

if __name__ == '__main__':
    app.run(debug=True)