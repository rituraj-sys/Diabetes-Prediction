from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

with open('D:\\conda3\\model\\abc.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()

    input_data = np.array([list(data.values())])

    prediction = model.predict(input_data)

    prediction_labels = ['non-diabetes', 'diabetes']
    predicted_label = prediction_labels[prediction[0]]

    return render_template('index.html', prediction_result=f"Result: {predicted_label}")

if __name__ == '__main__':
    app.run(debug=True)