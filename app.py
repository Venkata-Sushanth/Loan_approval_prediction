from flask import Flask, request, jsonify, send_from_directory
import joblib, json, numpy as np, os

app = Flask(__name__, static_folder='static')

MODEL_NAMES = ['gradient_boosting', 'random_forest', 'logistic_regression', 'knn_classifier']
models = {}
for name in MODEL_NAMES:
    path = f'model_{name}.pkl'
    if os.path.exists(path):
        models[name] = joblib.load(path)

meta = json.load(open('model_meta.json'))
feature_cols = meta['feature_cols']

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/meta')
def get_meta():
    return jsonify(meta)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        gender_map = {'Male': 1, 'Female': 0}
        married_map = {'Yes': 1, 'No': 0}
        edu_map = {'Graduate': 0, 'Not Graduate': 1}
        emp_map = {'Yes': 1, 'No': 0}
        area_map = {'Rural': 0, 'Semiurban': 1, 'Urban': 2}

        gender    = gender_map.get(data.get('Gender', 'Male'), 1)
        married   = married_map.get(data.get('Married', 'No'), 0)
        edu       = edu_map.get(data.get('Education', 'Graduate'), 0)
        self_emp  = emp_map.get(data.get('Self_Employed', 'No'), 0)
        area      = area_map.get(data.get('Property_Area', 'Urban'), 2)
        deps      = float(data.get('Dependents', 0))
        app_inc   = float(data.get('ApplicantIncome', 0))
        coapp_inc = float(data.get('CoapplicantIncome', 0))
        loan_amt  = float(data.get('LoanAmount', 0))
        term      = float(data.get('Loan_Amount_Term', 360))
        credit    = float(data.get('Credit_History', 1))

        total_income      = app_inc + coapp_inc
        log_total_income  = np.log1p(total_income)
        emi               = loan_amt / (term + 1)
        debt_income_ratio = emi / (total_income + 1)
        income_per_dep    = total_income / (deps + 1)
        loan_income_ratio = loan_amt / (total_income + 1)

        row = [
            gender, married, deps, edu, self_emp,
            app_inc, coapp_inc, loan_amt, term, credit, area,
            total_income, log_total_income, emi,
            debt_income_ratio, income_per_dep, loan_income_ratio
        ]

        import pandas as pd
        X = pd.DataFrame([row], columns=feature_cols)

        predictions = {}
        chosen_model = data.get('model', 'gradient_boosting')

        for name, pipe in models.items():
            prob = float(pipe.predict_proba(X)[0][1])
            pred = int(pipe.predict(X)[0])
            predictions[name] = {
                'probability': round(prob * 100, 1),
                'approved': pred == 1
            }

        chosen = predictions.get(chosen_model, list(predictions.values())[0])

        return jsonify({
            'status': 'ok',
            'approved': chosen['approved'],
            'probability': chosen['probability'],
            'all_predictions': predictions,
            'chosen_model': chosen_model
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    print("LoanWise running → http://localhost:5000")
    app.run(debug=True, port=5000)
