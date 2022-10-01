from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
import joblib

app = Flask(__name__)

# will use get for the first page-load, post for the form-submit
@app.route('/', methods=['post', 'get'])
def predict():  # this function can have any name
    # the nand.h5 file was created in Colab, downloaded and uploaded using Filezilla
    model = load_model('my_model.h5')
    scaler = joblib.load('scaler.gz')
    # get the two numbers from the request object
    in1 = request.form.get('in1')
    in2 = request.form.get('in2')
    in3 = request.form.get('in3')
    in4 = request.form.get('in4')
    in5 = request.form.get('in5')
    in6 = request.form.get('in6')
    in7 = request.form.get('in7')
    in8 = request.form.get('in8')
    in9 = request.form.get('in9')
    in10 = request.form.get('in10')
    in11 = request.form.get('in11')
    in12 = request.form.get('in12')
    in13 = request.form.get('in13')
    in14 = request.form.get('in14')
    in15 = request.form.get('in15')
    in16 = request.form.get('in16')
    if in1 == None or in2 == None:  # check if any number is missing
        return render_template('index.html', result="No input(s)")
        # calling render_template will inject the variable 'result' and send index.html to the browser
    else:
        arr = [[float(in1), float(in2), float(in3), float(in4), float(in5), float(in6), float(in7), float(in8), float(in9), float(
            in10), float(in11), float(in12), float(in13), float(in14), float(in15), float(in16)]] # cast string to decimal number, and make 2d numpy array.
        arr = scaler.transform(arr) 
        predictions = model.predict(arr)  # make new prediction
        return render_template('index.html', result=str(predictions[0][0]))
        # the result is set, by asking for row=0, column=0. Then cast to string.


if __name__ == '__main__':
    app.run()
