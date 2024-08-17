from flask import Flask, request, jsonify, render_template, redirect,   url_for, flash, send_file
import pickle
import pandas as pd
import xgboost as xgb
from flask_socketio import SocketIO, emit
import matplotlib.pyplot as plt

def plot(provider_id, X):
    print("Vanten")

    top_fraud_features = ['InscClaimAmtReimbursed','PerAttendingPhysicianAvg_InscClaimAmtReimbursed','PerOperatingPhysicianAvg_InscClaimAmtReimbursed']

    data = pd.read_csv('finalized_ds.csv')
    print("data loaded")
    provider_data = data[data['Provider'] == provider_id]
    print('1')
    overall_means = data[top_fraud_features].mean()
    print('2')
    provider_means = provider_data[top_fraud_features].mean()
    print('3')
    plt.figure(figsize=(10, 6))
    print('4')
    plt.bar(top_fraud_features, overall_means, label='All Providers', alpha=0.7)
    print('5')
    plt.bar(top_fraud_features, provider_means, label=f'Provider {provider_id}', alpha=0.7)
    print('6')
    #plt.set_facecolor("#AFE1AF")
    plt.xlabel('Fraud Indicator Features')
    plt.ylabel('Mean Value')
    plt.title(f'Fraud Indicators for Provider {provider_id}')
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('static/fraud_indicators.png')
    print('7')

    explanation = f"Provider {provider_id} shows "
    provider_data = data[data['Provider'] == provider_id]
    print("8")
    overall_means = data[top_fraud_features].mean()
    print("9")
    provider_means = provider_data[top_fraud_features].mean()
    print("10")
    for feature in top_fraud_features:
        if provider_means[feature] > overall_means[feature]:
            explanation += f"higher {feature}, "
        else:
            explanation += f"lower {feature}, "
    print("11")
    explanation = explanation[:-2]  
    explanation += " compared to the average of all providers."
    provider_features = provider_data[X.columns].values  
    print("12")

    # with open('Random_Forest_Finalized.pkl', 'rb') as file:
    #     model1 = pickle.load(file)
    fraud_prediction = model1.predict(provider_features)[0] 

    print("13")
    if fraud_prediction == 1:
        explanation += " This, combined with other factors, suggests a higher likelihood of fraudulent activity."
    else:
        explanation += " However, based on the overall profile, the model does not predict a high likelihood of fraud."
    
    return explanation


app = Flask(__name__)
app.secret_key = "InsuranceGuard"
socketio = SocketIO(app, cors_allowed_origins="*")
@socketio.on('message')
def handle_message(data):
    user_input = data['message']
    messages = [
    {"role": "user", "content": user_input},]
    pipe = pipeline("text-generation", model="microsoft/DialoGPT-medium",max_length=256)
    res = pipe(user_input)
    emit('response', {'message': user_input, 'generated_text': res[0]})


with open('Random_Forest_Finalized.pkl', 'rb') as file:
        model1 = pickle.load(file)

with open('Logistics_Regression_Finalized.pkl', 'rb') as file:
    model2 = pickle.load(file)

with open('RG_Boost_Finalized.pkl', 'rb') as file:
    model3 = pickle.load(file)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/find', methods=['POST'])
def fraud():
    provider_id = request.form.get('provider_id')
    print(provider_id)
    try:
        data = pd.read_csv('finalized_ds.csv')
        print("dataset loaded")
    
        i = data[data['Provider'] == provider_id].index.tolist()[0]
        print(i)
        X=data.drop(columns=['Provider','PotentialFraud'],axis=1)
        y=data['PotentialFraud']
        print("columns loaded")
        X = X[i:i+1]
        y = y[i:i+1]
        res = {}
        print("Done1")

        y_pred1 = model1.predict_proba(X)[0][1]*100
        res['Decision Tree'] = y_pred1
        print("The Percentage of being Fraud is", y_pred1)
        print("The Truth is",y)
        print("Done2")

        y_pred2 = model2.predict_proba(X)[0][1]*100
        res['Logistic Regression'] = y_pred2
        print("The Percentage of being Fraud is", y_pred2)
        print("The Truth is",y)
        print("Done3")

        drow = xgb.DMatrix(X)
        y_pred3 = model3.predict(drow)[0]*100
        res['XGBoost'] = y_pred3
        print("The Percentage of being Fraud is",y_pred3)
        print("Vela Completed")
        print("Done4")
        
        exp = plot(provider_id, X)
        return render_template('result.html', results=res, explanation = exp)

    except Exception as e:
        print("Error")
        return render_template('Error.html')

if __name__ == '__main__':
    socketio.run(app, debug=True)
