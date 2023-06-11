
import numpy as np
from flask import Flask, request, jsonify, render_template,session
import pickle
from collections import OrderedDict

from chatbot import ChatBot



app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
bot = ChatBot(model)
app.secret_key = "cscklsjcklsc5626223"

@app.route('/')
def home():
    return render_template('index-1.html')

@app.route('/prediction')
def prediction():
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/virtual_assistance')
def virtual_assistance():
    return render_template('virtual_assistance.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data['message']

    if 'predict' in message.lower():
        session['predict_mode'] = True
        session['features'] = ['LotArea', 'OverallQual', 'TotalBsmtSF', 'GarageArea', 'GrLivArea']
        session['feature_index'] = 0
        session['feature_values'] = OrderedDict()
        response = "Please enter the value of " + session['features'][session['feature_index']] + ":"
    elif session.get('predict_mode'):
        feature_value = float(message)
        session['feature_values'][session['features'][session['feature_index']]] = feature_value
        session['feature_index'] += 1
        if session['feature_index'] < len(session['features']):
            response = "Please enter the value of " + session['features'][session['feature_index']] + ":"
        else:
            ordered_values = OrderedDict((key, session['feature_values'][key]) for key in session['features'])
            print(ordered_values)  # Print the ordered dictionary
            response = bot.chat('predict', ordered_values)
            session.pop('predict_mode')
            session.pop('features')
            session.pop('feature_values')
            session.pop('feature_index')
    else:
        response = bot.chat(message, None)

    return jsonify({'message': response})




@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='The house price is $ {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)



if __name__ == "__main__":
    app.run(debug=True)
