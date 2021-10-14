from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def hello():
    return 'hello, world'

@app.route('/tts')
def retrieve_data():
    input_text = request.args.get('input')
    emotion_tag = request.args.get('emotion')
    
    return input_text + " and " + emotion_tag

if __name__ == "__main__":
    app.run(debug=True)