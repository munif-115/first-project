import numpy as np
from flask import Flask,request,render_template
import pickle
from werkzeug.serving import run_simple

app=Flask(__name__,template_folder='template')
model=pickle.load(open("model.pkl",'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    arr=[int(x) for x in request.form.values()]
    arr2=[np.array(arr)]
    output=model.predict(arr2)
   # o2=round(output)
    return render_template('index.html',prediction_text=output)










if __name__ == "__main__":
    run_simple('localhost',8001,app,use_reloader=False)