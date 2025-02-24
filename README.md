# Player Score Prediction

This project uses XGBoost to predict if a player will score above a certain betline in a game based on historical data.
XGBoost regression model is used predict the number of points a player is likely to score in their next game. This prediction is based on historical data, rolling averages, and trends. The model outputs a numerical value representing the expected points.

Once the prediction is made:

It can be compared against a betting line (a predefined threshold set by bookmakers for a player's performance).
This comparison helps decide whether the player is likely to score "over" or "under" that line, aligning the regression output with the binary outcome commonly needed for sports betting decisions.

This combination of regression and decision-making relative to a threshold essentially bridges continuous predictions with actionable insights for the betting scenario.

## How to Start the Project

### Prerequisites

- Ensure you have Python installed (preferably Python 3.8+).
- Install the required Python libraries using the following command:
  ```bash
  pip install -r requirements.txt
  ```
  Use the provided commands.py file to scrape the player data.
  Run the following command to execute the scraping process:

Steps

Scrape Data
Use the provided cli_app.py file to scrape the player data.
Run the following command to execute the scraping process:

```bash
python cli_app.py scrape-init
```

This will:

    Call the scrape_seasons() function to scrape data for specific seasons.
    Call the scrape_team_seasons() function to scrape team-level data for the same seasons.

Important:
Ensure that the arrays of seasons defined in both scrape_seasons and scrape_team_seasons are consistent, as they must cover the same range of data.

Run the Model
After data preparation, you can proceed to train or test the XGBoost model. Instructions for this are provided in the relevant script documentation.

```bash
python cli_app.py train-xgb
```

## Making Predictions

### Input Data

To make predictions, you need to provide input data in a JSON file named `data.json`. This file should contain an array of objects, where each object has the following structure:

```json
[
  {
    "name": "Player Name",
    "points": 25
  },
  {
    "name": "Another Player",
    "points": 18
  }
]
```

Running Predictions

Once the data.json file is populated with the required data, you can run the following command to make predictions:

```bash
python cli_app.py predict-all
```

In order to display predictions there is a flask application where u can see predictions made for the day
flask application is very simple feel free to change the query to adhere your needs for showing predictions

```
flask run --debug
```

### Project Limitations and Maintenance

This project is a basic MVP (Minimum Viable Product) designed as a hobby project and relies heavily on manual commands to maintain and synchronize data. It is not fully automated, so keeping the data accurate and up-to-date requires regular intervention.

To ensure the predictions remain relevant:

- **Run the scraping mechanism daily** using the command:
  ```
  python cli_app.py fill-data
  ```

This is necessary because NBA games typically occur daily or every other day.

While the project demonstrates the concept effectively, it is not yet robust or production-ready. Regular updates and maintenance are required to keep the data and predictions accurate.
