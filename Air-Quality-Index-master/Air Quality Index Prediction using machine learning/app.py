# IMPORT LIBRARIES
import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle

# CREATING A FLASK APP
app = Flask(__name__)

# LOAD THE PICKLE MODEL
model = pickle.load(open("linear_model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/", methods=["POST"])
def predict():
    T = float(request.form['Temperature(T)'])
    TM = float(request.form['Maximum Temperature(TM)'])
    Tm = float(request.form['Tm'])
    SLP = float(request.form['Atmoshperic pressure at sea level (SLP) in hPa'])
    H = float(request.form['Average Relative Humidity(H) in(%)'])
    VV = float(request.form['Average Visbility(VV) in (Km)'])
    V = float(request.form['V'])
    VM = float(request.form['VM'])
    features_list = [T, TM, Tm, SLP, H, VV, V, VM]
    features = [np.array(features_list)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text="The Air Quality Index (AQI) is {}".format(prediction))


if __name__ == "__main__":
    app.run(debug=True)
