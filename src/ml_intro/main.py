"""Main entry point for ml_intro."""
from pathlib import Path
from sklearn.tree import DecisionTreeRegressor

import pandas as pd


def main() -> None:
    """Run the main application."""
    melburn_data_path = (
        Path(__file__).resolve().parent.parent.parent
        / "datasets"
        / "melbourne-housing-snapshot"
        / "melb_data.csv"
    )
    melbourne_data = pd.read_csv(melburn_data_path)
    y = melbourne_data.Price
    summary_table = melbourne_data.describe()
    print("Summary table for Melbourne housing data:")
    print(summary_table)

    print("\nColumn names in Melbourne housing data:")
    print(melbourne_data.columns)

    melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
    X = melbourne_data[melbourne_features]
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


if __name__ == "__main__":
    main()
