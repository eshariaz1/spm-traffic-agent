# dataset_loader.py

import pandas as pd
from typing import List, Dict


def load_dataset(path: str = "traffic_data.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def filter_dataset(df, scenario: str, intersections: List[str]):
    # match rows where scenario matches AND intersections overlap
    filtered = df[df["scenario"] == scenario]

    # keep rows where ANY intersection matches
    result = []
    for _, row in filtered.iterrows():
        row_intersections = str(row["intersections"]).split(",")
        row_intersections = [i.strip() for i in row_intersections]

        if any(i in row_intersections for i in intersections):
            result.append(row)

    return pd.DataFrame(result)


def aggregate_rows(df):
    if df.empty:
        return None

    return {
        "total_volume": int(df["volume"].sum()),
        "average_queue_length": float(df["queue_length"].mean()),
        "average_speed": float(df["avg_speed"].mean()),
        "num_records": int(len(df)),
        "max_queue_length": int(df["queue_length"].max())
    }

