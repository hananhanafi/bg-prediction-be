from flask import Flask, request, jsonify
import json
import pickle
import psycopg2
from flask_cors import CORS, cross_origin
import numpy as np

app = Flask(__name__)
CORS(app, support_credentials=True)
model = pickle.load(open('model.pkl','rb'))

def get_db_connection():
    conn = psycopg2.connect("postgresql://postgres:112233@localhost:5432/db_bg_prediction")
    return conn

@app.route("/")
def index():
    return "Welcome to BG Level Prediction APIs!"

@app.route('/patients', methods=['GET'])
def patients():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT * FROM patient_records order by ptid,bgdatetime;')
    patient_records = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify({'data': patient_records})  # return model output

@app.route('/patients_id', methods=['GET'])
def patient_id():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('select pr.ptid from patient_records pr group by pr.ptid order by pr.ptid;')
    patient_records = cur.fetchall()
    cur.close()
    conn.close()
    arr = np.array(patient_records)
    result = arr.flatten()
    result_str = [int(i) for i in result]  # save the result
    return jsonify({'data': result_str})  # return model output

@app.route('/patient/<int:id>', methods=['GET'])
def patient(id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT * FROM patient_records where ptid = ' + str(id) + ' order by ptid,bgdatetime;')
    patient_records = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify({'data': patient_records})  # return model output

@app.route('/patient/predict', methods=['POST'])
def get_prediction():
    req_json = json.loads(request.data)  # read request
    ph = int(req_json['ph'])
    data = req_json['data']
    x = data
    
    result = []
    for i in range(ph):
        result = model.predict(x)  # predict
        x[0].pop(0)
        x[0].append(result[0])
    result_str = [(i) for i in result]  # save the result
    return jsonify({'data': result_str})  # return model output

if __name__ == '__main__':
    app.run(debug=True)