import pandas as pd
from flask import Flask, jsonify, request
import pickle
#load model
model = pickle.load(open('model.pkl', 'rb'))
print(model)

# app 
app = Flask (__name__)
# routes
@app.route('/models', methods=['POST'])
def predict():
# get data

    data = request.get_json()
    print(data)
    # convert data into dataframe
    # data.update((x, [y]) for x, y in data.items()) 
    # data_df = pd.DataFrame.from_dict(data)
    # predictions
    result= model.predict([[data['hours_worked']]])
    #send back to browser
    result={"predicted_score":result[0]}
    # return data
    return jsonify(result)


if __name__ == '__main__':
    app.run(port=5000,debug=True)