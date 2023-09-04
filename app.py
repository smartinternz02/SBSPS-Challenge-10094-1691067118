import numpy as np
from flask import Flask, render_template, request
import pickle
from sklearn.impute import SimpleImputer

app = Flask(__name__, template_folder="templates")
model = pickle.load(open('yamu.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["GET", "POST"])
def predict():
    ssc_p = float(request.form.get('ssc_p'))
    hsc_p = float(request.form.get('hsc_p'))
    degree_p = float(request.form.get('degree_p'))
    etest_p = float(request.form.get('etest_p'))
    mba_p = float(request.form.get('mba_p'))

    
    arr = np.array([[ssc_p, hsc_p,  degree_p,  etest_p, mba_p]])
    brr=np.asarray(arr,dtype=float)
    output = model.predict(arr)
    
    if output == 1:
        out = 'You have high chances of getting placed!!!'
    else:
        out = 'You have low chances of getting placed. All the best.'
        
    return render_template('out.html', output=out) 

if __name__ == '__main__':
    app.run(debug=True)