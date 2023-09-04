import numpy as np
from flask import Flask, render_template, request
import requests

app = Flask(__name__)

# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "<cbb5c546-1090-44b2-a7f5-007cdfbccfbe>"

# Add error handling for the token request
try:
    token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
    mltoken = token_response.json().get("access_token")
    if mltoken is None:
        raise ValueError("Access token not found in the response.")
except Exception as e:
    print("Error retrieving access token:", str(e))
    mltoken = None

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken if mltoken else ''}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["GET", "POST"])
def predict():
    if mltoken is None:
        return "Error: Unable to get access token. Please check your API_KEY."

    ssc_p = float(request.form.get('ssc_p'))
    hsc_p = float(request.form.get('hsc_p'))
    degree_p = float(request.form.get('degree_p'))
    etest_p = float(request.form.get('etest_p'))
    mba_p = float(request.form.get('mba_p'))

    arr = np.array([[ssc_p, hsc_p,  degree_p,  etest_p, mba_p]])
    brr = np.asarray(arr, dtype=float)

    # NOTE: manually define and pass the array(s) of values to be scored in the next line
    payload_scoring = {"input_data": [{"fields": [ssc_p, hsc_p,  degree_p,  etest_p, mba_p], "values": [arr]}]}

    response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/6bfdaa4f-cd0c-4b41-9165-99c6d3bf1e6c/predictions?version=2021-05-01', json=payload_scoring, headers={'Authorization': 'Bearer ' + mltoken})
    
    if response_scoring.status_code != 200:
        return "Error: Failed to make a prediction. Status code: {}".format(response_scoring.status_code)

    predictions = response_scoring.json()
    output = predictions['predictions'][0]['value'][0][0]

    if output == 1:
        out = 'You have high chances of getting placed!!!'
    else:
        out = 'You have low chances of getting placed. All the best.'

    return render_template('out.html', output=out)

if __name__ == '__main__':
    app.run(debug=True)
