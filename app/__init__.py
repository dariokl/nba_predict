
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


def get_predictions(date, page, items_per_page):
    """Fetch paginated predictions from the database."""
    offset = (page - 1) * items_per_page

    with sq.connect(db_path) as conn:
        cursor = conn.cursor()

        # Get total item count
        cursor.execute(
            """
            SELECT COUNT(*) 
            FROM predictions
            WHERE type = 'trend' AND DATE(date) = ?
            """,
            (date,)
        )
        total_items = cursor.fetchone()[0]
        total_pages = ceil(total_items / items_per_page)

        # Fetch paginated predictions
        cursor.execute(
            """
            SELECT json_object(
                'player_name', player_name,
                'betline', betline,
                'over_under', over_under,
                'predicted_points', predicted_points,
                'win', win,
                'scored_points', scored_points,
                'confidence', confidence
            ) 
            FROM predictions
            WHERE type = 'trend' AND DATE(date) = ?
            ORDER BY confidence DESC
            LIMIT ? OFFSET ?
            """,
            (date, items_per_page, offset)
        )
        predictions = [json.loads(row[0]) for row in cursor.fetchall()]

    return predictions, total_pages


def create_app():
    """Flask app factory function."""
    app = Flask(__name__)

    # Asset bundling
    assets = Environment(app)
    css = Bundle("src/main.css", output="dist/main.css")
    assets.register("css", css)
    css.build()

    @app.route("/")
    def homepage():
        """Render the homepage."""
        return render_template("index.html")

    @app.route('/predictions')
    def predictions():
        """Fetch and return paginated predictions."""
        try:
            today = (datetime.today()).strftime('%Y-%m-%d')
            page = int(request.args.get('page', 1))
            date = request.args.get('date', today)

            # Validate date format
            try:
                date = datetime.strptime(date, '%Y-%m-%d').strftime('%Y-%m-%d')
            except ValueError:
                return {"error": "Invalid date format. Use YYYY-MM-DD"}, 400

            items_per_page = 10
            predictions, total_pages = get_predictions(
                date, page, items_per_page)

            return {
                "predictions": predictions,
                "page": page,
                "total_pages": total_pages,
            }

        except Exception as e:
            return {"error": str(e)}, 500

    return app
