from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.preprocessing import RobustScaler
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

rbs = RobustScaler()

@app.route('/')
def home():
   return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
   file = request.files['file']
   df = pd.read_csv(file)
   df_small = df[['Time','Amount']]
   rbs.fit(df_small)
   df_small = pd.DataFrame(rbs.transform(df_small))
   df_small.columns = ['scaled_time','scaled_amount']
   df.drop(['Time','Amount', 'Class'],axis=1,inplace=True)
   df = pd.concat([df,df_small],axis=1)
   predictions = model.predict(df)
   fraud_transactions = df[predictions == 1]
   # print(fraud_transactions)
   print(fraud_transactions.columns)
   return render_template('result.html', tables=[fraud_transactions.to_html(classes='data')], titles=fraud_transactions.columns.values)


if __name__ == '__main__':
   app.run(debug=True)
