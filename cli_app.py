import typer
from app.actions.model import train_model_and_save_model_xgboost, train_model_and_save_model_gradient
from app.actions.predictions import predict_from_json, fill_win_column, predictions_stats
from app.actions.scrape import scrape_seasons, scrape_team_seasons
from app.actions.db_operations import fill_data_to_db
from app.actions.model import backtest

app = typer.Typer()


@app.command()
def train_xgb():
    """Train the model."""
    train_model_and_save_model_xgboost()


@app.command()
def train_gradient():
    """Train the model."""
    train_model_and_save_model_gradient()


@app.command()
def predict_all():
    """Run all predictions."""
    predict_from_json('mean')
    predict_from_json('trend')


@app.command()
def predict_mean():
    """Run mean prediction."""
    predict_from_json('mean')


@app.command()
def predict_trend():
    """Run trend prediction."""
    predict_from_json('trend')


@app.command()
def scrape_init():
    """Scrape data."""
    scrape_seasons()
    scrape_team_seasons()


@app.command()
def fill_predictions():
    """Fill predictions table."""
    fill_win_column()
    predictions_stats()


@app.command()
def get_predictions_stats():
    """Generate prediction statistics."""
    predictions_stats()


@app.command()
def fill_data():
    """Fill data to the database."""
    fill_data_to_db()


@app.command()
def backtest_predictions():
    """Run backtesting."""
    backtest()


if __name__ == "__main__":
    app()
