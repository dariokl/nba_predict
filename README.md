# Player Score Prediction

This project uses XGBoost to predict if a player will score above a certain betline in a game based on historical data and rolling averages.

## Features

- **Train the Model**: Trains an XGBoost model on player stats.
- **Make Predictions**: Predicts if a player will exceed a specified point betline.
- **Command-line Interface**: Use `train` to train the model or `predict` to make predictions. Use `scrape` to create data the csv that model is using to train.

## Requirements

- Python 3.x
- Libraries: `xgboost`, `pandas`, `scikit-learn`

Install dependencies:

```bash
pip install xgboost pandas scikit-learn
```

Train the Model

```bash
python script_name.py train
```

Make Predictions

```bash
python script_name.py predict
```

Get Latest Player Data

```bash
python script_name.py scrape
```

## Input format

Inside data.json check the input that is required to perform predictions.
