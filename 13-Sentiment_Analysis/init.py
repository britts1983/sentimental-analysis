from flask import Flask, render_template, redirect, flash,request
import sentiment_mod as s

app = Flask(__name__)
app.secret_key = "TeddyWinters"

@app.route('/', methods=['GET','POST'])
def home():
    text=""
    result = ""
    confidence=0
    try:
        if request.method == 'POST':
            text = str(request.form['review'])
            result, confidence = s.sentiment(text)
            if result == "pos":
                result = "positive"
            else:
                result = "negative"

    except Exception as e:
        print e

    return render_template('index.html',text=text,result=result, confidence=confidence)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8000)
    app.debug=True