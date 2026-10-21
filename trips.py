import pandas as pd
# add_pickup_day: the day of each trip, as `2024-01-15`
def add_pickup_day(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pickup_day"] = df["tpep_pickup_datetime"].dt.strftime("%Y-%m-%d")
    return df
# tip_percentage: how much people tip, as a percentage of the fare
def tip_percentage(df: pd.DataFrame) -> pd.Series:
    paid = df[df["fare_amount"] > 0]
    return paid["tip_amount"] * 100 / paid["fare_amount"]