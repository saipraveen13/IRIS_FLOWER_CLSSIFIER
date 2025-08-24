from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.datasets import load_iris

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))
iris = load_iris()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    flower = iris.target_names[prediction]

    return render_template('index.html', prediction_text=f"🌼 Predicted Species: {flower}")

if __name__ == "__main__":
    app.run(debug=True)
