from flask import Flask, render_template, request, redirect, url_for


app = Flask(__name__)


@app.before_request
def method_override():
    if request.method == 'POST' and '_method' in request.form:
        method = request.form['_method'].upper()
        if method in ['PUT', 'DELETE', 'PATCH']:
            request.environ['REQUEST_METHOD'] = method


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', '').lower()
    results = data.get(query, [])
    return render_template('results.html', results=results)


if __name__ == '__main__':
    app.run(debug=True)
