from flask import Flask, render_template, request
import numpy as np
import pickle
import joblib
app = Flask(__name__)
filename = 'file_ethnicity.pkl'
#model = pickle.load(open(filename, 'rb'))
model = joblib.load(filename)
#model = joblib.load(filename)
@app.route('/')
def index(): 
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    math_score = float(request.form['math_score'])
    reading_score = float(request.form['reading_score'])
    writing_score = float(request.form['writing_score'])

    
      
    pred = model.predict(np.array([[math_score, reading_score, writing_score]]))
    print(pred)
    return render_template('index.html', predict=str(pred))


if __name__ == '__main__':
    app.run
