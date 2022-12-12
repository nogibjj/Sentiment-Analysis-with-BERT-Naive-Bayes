"""Data access utilities."""
import csv
from pathlib import Path
import pandas as pd

FILEPATH = Path(__file__).parent
IMDB_FILENAME = FILEPATH / "imdb_master.csv"


def get_imdb_data():
    """Get IMDB data."""
    with open(IMDB_FILENAME, "r") as stream:
        data = pd.read_csv(IMDB_FILENAME, encoding="ISO-8859-1")
    return data


if __name__ == "__main__":
    print(get_imdb_data())
