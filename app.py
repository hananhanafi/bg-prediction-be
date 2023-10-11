from flask import Flask, request, jsonify
import json
import pickle
import psycopg2
from flask_cors import CORS, cross_origin
import numpy as np
from scipy.stats import kurtosis, skew

app = Flask(__name__)
CORS(app, support_credentials=True)

model = pickle.load(open('./model.pkl','rb'))

def get_db_connection():
    conn = psycopg2.connect("postgres://default:YHaCZebV0w7W@ep-sparkling-queen-33979936.ap-southeast-1.postgres.vercel-storage.com:5432/verceldb")
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
    result_str = [int(i) for i in result]  
    return jsonify({'data': result_str})  

@app.route('/patient/<int:id>', methods=['GET'])
def patient(id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT ptid, bgdatetime, bglevel FROM patient_records where ptid = ' + str(id) + ' order by ptid,bgdatetime;')
    patient_records = cur.fetchall()
    cur.close()
    conn.close()
    result_str = [{
                "pt_id": i[0],
                "bg_datetime": i[1],
                "bg_level": i[2]
            } for i in patient_records]  
    return jsonify({'data': result_str})  

def stats_features(input_data):
    inp = list()
    for i in range(len(input_data)):
        inp2=list()
        inp2=input_data[i]
        min=float(np.min(inp2))
        max=float(np.max(inp2))
        diff=(max-min)
        std=float(np.std(inp2))
        mean=float(np.mean(inp2))
        median=float(np.median(inp2))
        kurt=float(kurtosis(inp2))
        sk=float(skew(inp2))
        inp2=np.append(inp2,min)
        inp2=np.append(inp2,max)
        inp2=np.append(inp2,diff)
        inp2=np.append(inp2,std)
        inp2=np.append(inp2,mean)
        inp2=np.append(inp2,median)
        inp2=np.append(inp2,kurt)
        inp2=np.append(inp2,sk)
        inp2 = np.nan_to_num(inp2)
        inp=np.append(inp,inp2)
    inp=inp.reshape(len(input_data),-1)
    return inp

@app.route('/patient/predict', methods=['POST'])
def get_prediction():
    req_json = json.loads(request.data)  # read request
    ph = int(req_json['ph'])
    data = req_json['data']
    x = data
    
    resultArr = []
    result = []
    for i in range(ph):
        x_with_stats = stats_features(x) # extract stats feature
        result = model.predict(x_with_stats)  # predict
        x[0].pop(0)
        x[0].append(result[0])
        resultArr.append(result[0])
    result_str = [(i) for i in resultArr]  # save the result
    return jsonify({'data': result_str})  # return model output

if __name__ == '__main__':
    app.run(debug=True)