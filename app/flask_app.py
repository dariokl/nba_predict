
from flask import Flask, render_template
from flask_assets import Bundle, Environment
import os
import csv

import pandas as pd

app = Flask(__name__)

assets = Environment(app)
css = Bundle("src/main.css", output="dist/main.css")

assets.register("css", css)
css.build()


@app.route("/")
def homepage():
    csv_file = os.path.join(os.path.dirname(__file__),
                            '../', 'predictions_2024-11-18_trend.csv')
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(rows)
    return render_template("index.html", predictions=rows)


if __name__ == "__main__":
    app.run(debug=True)
