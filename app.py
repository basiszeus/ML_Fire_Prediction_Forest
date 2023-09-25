import joblib
from flask import Flask, request, render_template
import numpy as np
from app_logger import log

# Load the models
model_R = joblib.load('best_regressor_model.pkl')

app = Flask(__name__)

# Route for homepage
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predictR', methods=['POST'])
def predictR():
    if request.method == 'POST':
        try:
            # Read the inputs given by the user Regression
            Temperature = float(request.form['Temperature'])
            RH = int(request.form['RH'])
            Wind_Speed = int(request.form['Ws'])
            FFMC = float(request.form['FFMC'])
            DMC = float(request.form['DMC'])
            ISI = float(request.form['ISI'])
            Rain = float(request.form['Rain'])

            featuresr = [Temperature, RH, Wind_Speed, FFMC, DMC, ISI, Rain]

            Float_features = [float(x) for x in featuresr]
            final_features = [np.array(Float_features)]
            prediction = model_R.predict(final_features)[0]

            if prediction > 15:
                return render_template('index.html', prediction_text2="Fuel Moisture Code index is {:.4f} ---- Warning!!! High hazard rating".format(prediction))
            else:
                return render_template('index.html', prediction_text2="Fuel Moisture Code index is {:.4f} ---- Safe.. Low hazard rating".format(prediction))
        except Exception as e:
            log.error('Input error, check input', e)
            return render_template('index.html', prediction_text2="Check your Input again")

if __name__ == '__main__':
    app.run(debug=True, port=8000)