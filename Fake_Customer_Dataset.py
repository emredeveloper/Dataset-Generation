"""Utilities for generating synthetic customer datasets.

This module provides a command-line interface that can be used to create
reproducible synthetic customer data suitable for demos, experimentation or
bootstrapping analytics pipelines.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from faker import Faker
from sklearn.preprocessing import MinMaxScaler


def generate_customer_data(
    num_records: int = 1000,
    *,
    seed: Optional[int] = None,
    locale: str = "en_US",
    normalize_numeric: bool = True,
) -> pd.DataFrame:
    """Generate a synthetic customer dataset.

    Parameters
    ----------
    num_records:
        Number of synthetic customer entries to create.
    seed:
        Optional random seed to make the generated dataset deterministic.
    locale:
        Locale to initialise the Faker instance with.
    normalize_numeric:
        When ``True`` the ``Annual_Income`` and ``Purchase_Amount`` columns are
        min-max normalised to the 0-1 range.
    """
    if num_records <= 0:
        raise ValueError("'num_records' must be a positive integer")

    faker = Faker(locale)
    if seed is not None:
        Faker.seed(seed)
        faker.seed_instance(seed)
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    data = {
        "Customer_ID": [faker.uuid4() for _ in range(num_records)],
        "Name": [faker.name() for _ in range(num_records)],
        "Age": rng.integers(18, 80, size=num_records),
        "Gender": rng.choice(["Male", "Female", "Other"], size=num_records),
        "Email": [faker.email() for _ in range(num_records)],
        "Phone": [faker.phone_number() for _ in range(num_records)],
        "Country": [faker.country() for _ in range(num_records)],
        "Signup_Date": [faker.date_between(start_date="-5y", end_date="today") for _ in range(num_records)],
        "Annual_Income": rng.integers(20_000, 150_000, size=num_records),
        "Purchase_Amount": rng.uniform(10, 5_000, size=num_records),
        "Loyalty_Score": rng.uniform(0, 1, size=num_records),
    }

    df = pd.DataFrame(data)

    if normalize_numeric:
        scaler = MinMaxScaler()
        df[["Annual_Income", "Purchase_Amount"]] = scaler.fit_transform(
            df[["Annual_Income", "Purchase_Amount"]]
        )

    df["Purchase_Amount"] = df["Purchase_Amount"].round(2)
    df["Loyalty_Score"] = df["Loyalty_Score"].round(2)

    return df


def generate_summary_report(df: pd.DataFrame) -> str:
    """Create a textual summary for the provided dataset."""
    numeric_cols = df.select_dtypes(include=[np.number])
    categorical_cols = df.select_dtypes(exclude=[np.number])

    sections: list[str] = []

    if not numeric_cols.empty:
        sections.append("Numeric column summary:\n" + numeric_cols.describe().round(2).to_string())

    for column in categorical_cols.columns:
        value_counts = df[column].value_counts().head(5)
        sections.append(
            f"Top values for '{column}':\n" + value_counts.to_string()
        )

    return "\n\n".join(sections)


def save_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """Persist the dataset to disk as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic customer dataset.")
    parser.add_argument(
        "-n",
        "--records",
        type=int,
        default=1000,
        help="Number of synthetic customer records to generate (default: 1000).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional path to save the generated dataset as CSV.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed used to make the random number generation deterministic.",
    )
    parser.add_argument(
        "--locale",
        default="en_US",
        help="Locale identifier used by Faker (default: en_US).",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable min-max normalisation for numeric spend columns.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print a short exploratory data analysis report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = generate_customer_data(
        num_records=args.records,
        seed=args.seed,
        locale=args.locale,
        normalize_numeric=not args.no_normalize,
    )

    print("Örnek Kayıtlar:")
    print(df.head())

    if args.report:
        print("\nÖzet Rapor:")
        print(generate_summary_report(df))

    if args.output:
        save_dataset(df, args.output)
        print(f"\nVeri kümesi '{args.output}' dosyasına kaydedildi.")


if __name__ == "__main__":
    main()
