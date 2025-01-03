from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained Random Forest model
model = joblib.load('loan_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_data = [
        data['Gender'], data['Married'], data['Dependents'],
        data['Education'], data['Self_Employed'], data['ApplicantIncomelog'],
        data['LoanAmountlog'], data['Loan_Amount_Term_log'],
        data['Credit_History'], data['Property_Area']
    ]
    prediction = model.predict([input_data])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
