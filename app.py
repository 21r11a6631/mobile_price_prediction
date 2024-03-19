from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and DataFrame
data = pickle.load(open('data2.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index1.html', ppi=data['ppi'].unique(), cpu_core=data['cpu core'].unique(),
                           cpu_freq=data['cpu freq'].unique(), internal_mem=data['internal mem'].unique(),
                           ram=data['ram'].unique(), rear_cam=data['RearCam'].unique(),
                           front_cam=data['Front_Cam'].unique(), battery=data['battery'].unique(),
                           thickness=data['thickness'].unique(),weight=data['weight'].unique(),resolution=data['resoloution'].unique())

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve form data
        ppi = int(request.form['ppi'])
        cpu_core = int(request.form['cpu_core'])
        cpu_freq = float(request.form['cpu_freq'])
        internal_mem = float(request.form['internal_mem'])
        ram = float(request.form['ram'])
        rear_cam = float(request.form['rear_cam'])
        front_cam = float(request.form['front_cam'])
        battery = int(request.form['battery'])
        thickness = float(request.form['thickness'])
        weight = float(request.form['weight'])
        resolution = float(request.form['resolution'])

        model = pickle.load(open('model.pkl', 'rb'))

        # Create query array
        query = np.array([ppi, cpu_core, cpu_freq, internal_mem, ram, rear_cam, front_cam, battery, thickness, weight, resolution])
        query = query.reshape(1, -1)

        # Predict the price
        predicted_price = model.predict(query)[0]

        return render_template('index1.html', predicted_price=predicted_price)
if __name__ == '__main__':
    app.run(debug=True)
