"""Main entry point for ml_intro."""
from pathlib import Path

from src.ml_intro.utility import get_mae
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


def main() -> None:
    """Run the main application."""
    melburn_data_path = (
        Path(__file__).resolve().parent.parent.parent
        / "datasets"
        / "melbourne-housing-snapshot"
        / "melb_data.csv"
    )
    melbourne_data = pd.read_csv(melburn_data_path)

    # Filter rows with missing price values
    filtered_melbourne_data = melbourne_data.dropna(axis=0)

    y = filtered_melbourne_data.Price
    summary_table = melbourne_data.describe()
    print("Summary table for Melbourne housing data:")
    print(summary_table)

    print("\nColumn names in Melbourne housing data:")
    print(melbourne_data.columns)

    melbourne_features = ['Rooms', 'Bathroom', 'Landsize',
                           'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']
    X = filtered_melbourne_data[melbourne_features]
    print("Data used to predict housing prices:")
    print(X.describe())

    print("\nFirst few rows of the data used to predict housing prices:")
    print(melbourne_data.head())

    melbourne_model = DecisionTreeRegressor(random_state=1)
    melbourne_model.fit(X, y)

    print("\nMaking predictions for the following 5 houses:")
    print(X.head())
    print("The predictions are")
    print(melbourne_model.predict(X.head()))

    #  Calculate MAE for the model
    predicted_home_prices = melbourne_model.predict(X)
    mae = mean_absolute_error(y, predicted_home_prices)
    print(f"\nMean Absolute Error for the model: {mae}")

    # split data into training and validation data, for both features and target
    # The split is based on a random number generator. Supplying a numeric value to
    # the random_state argument guarantees we get the same split every time we
    # run this script.
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
    # Define model
    melbourne_model = DecisionTreeRegressor(random_state=1)
    # Fit model
    melbourne_model.fit(train_X, train_y)
    # Get predicted prices on validation data
    val_predictions = melbourne_model.predict(val_X)
    # Calculate MAE
    val_mae = mean_absolute_error(val_y, val_predictions)
    print(f"\nValidation MAE: {val_mae}")

    for max_leaf_nodes in [5, 50, 500, 5000]:
        my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
        print(f"Max leaf nodes: {max_leaf_nodes} \t\t Mean Absolute Error: {my_mae}")



if __name__ == "__main__":
    main()
