
from flask import Flask,render_template,request
from flask.wrappers import Request
from vocabulary.vocab import *
import dill


model_address = 'machine_learning_model/sentiment_analysis_prototype_1.pkl'
with open(model_address,'rb') as f : 
    M = dill.load(f)


app=Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/about/')
def about():
    return render_template("about.html")
 
@app.route('/predict/',methods=['POST'])
def predict() :
    if request.method == 'POST' :
        message = request.form['message']
        #check if we got some input
        if len(message) == 0 : return render_template('home.html', prediction= 'No message')
        statement = np.asarray([message])
        vector = corpus_to_feature_matrix_fast(statement,len(vocab['token_to_idx']) )
        sentiment_prediction = M.predict(vector)
        return render_template('result.html' , prediction = sentiment_prediction)
        
if __name__ == '__main__' :
    app.run(debug=True)
    