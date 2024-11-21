
import os
import sqlite3 as sq
import json
from datetime import datetime, timedelta
from math import ceil

from flask import Flask, render_template, request
from flask_assets import Bundle, Environment

from .config import Config


db_path = os.path.join(os.path.dirname(__file__),
                       '../', 'nba_predict.sqlite')


def create_app():
    # Consider adding blueprints as project grows
    app = Flask(__name__)
    app.config.from_object(Config)

    assets = Environment(app)
    css = Bundle("src/main.css", output="dist/main.css")

    assets.register("css", css)
    css.build()

    @app.route("/")
    def homepage():

        return render_template(
            "index.html")

    @app.route('/predictions')
    def predictions():
        today = (datetime.today() - timedelta(days=0)
                 ).strftime('%Y-%m-%d')
        # Default to page 1 if no page is specified
        page = int(request.args.get('page', 1))
        items_per_page = 10
        offset = (page - 1) * items_per_page

        connection = sq.connect(db_path)
        cursor = connection.cursor()

        # Count total items for pagination logic
        count_query = """
            SELECT COUNT(*)
            FROM predictions
            WHERE type = 'trend' AND DATE(date) = ?
        """
        total_items = cursor.execute(count_query, (today,)).fetchone()[0]
        total_pages = ceil(total_items / items_per_page)

        # Query paginated items
        query = """
            SELECT json_object(
                'player_name', player_name,
                'betline', betline,
                'over_under', over_under,
                'predicted_points', predicted_points
            ) as item
            FROM predictions
            WHERE type = 'trend' AND DATE(date) = ?
            LIMIT ? OFFSET ?
        """
        rows = cursor.execute(
            query, (today, items_per_page, offset)).fetchall()

        predictions = [json.loads(row[0]) for row in rows]

        cursor.close()
        connection.close()

        context = {
            "predictions": predictions,
            "page": page,
            "total_pages": total_pages
        }

        return context

    return app
