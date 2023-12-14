from flask import Flask, render_template, request
from agent import agent_executor

app = Flask(__name__)


@app.route('/')
def form():
    return render_template('main.html')

@app.route('/ask', methods=['POST'])
def lookup():
    print(request.form)
    results = agent_executor(request.form["ask"])
    return render_template('results.html', question=request.form["ask"], answer=results["output"].split("\n"))

if __name__ == '__main__':
    app.run(host='0.0.0.0')
