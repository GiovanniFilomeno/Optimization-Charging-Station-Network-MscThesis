"""
Data Preprocessing Module.

Cleans and normalizes the raw Bundesnetzagentur EV Charging Station Registry CSV.

Processing steps:
    1. Rename German column headers to English
    2. Standardize charger types (Schnellladeeinrichtung → fast)
    3. Handle missing values in secondary plug/power columns
    4. Fix German decimal format (comma → dot)
    5. Parse commissioning dates
    6. Normalize city names (merge districts, fix encoding)
    7. Remove duplicates

Usage:
    python -m src.preprocessing
"""

import logging
from pathlib import Path

import pandas as pd

from .config import RAW_CSV, CLEANED_CSV, DATA_PROCESSED

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Column name mapping (German → English)
# ──────────────────────────────────────────────────────────────────────
COLUMN_MAPPING = {
    "Betreiber": "operator",
    "Straï¿½e": "address",
    "Hausnummer": "house_number",
    "Adresszusatz": "placeholder1",
    "Postleitzahl": "postcode",
    "Ort": "city",
    "Bundesland": "federal_state",
    "Kreis/kreisfreie Stadt": "metropolitan_area",
    "Breitengrad": "latitude_[dg]",
    "Lï¿½ngengrad": "longitude_[dg]",
    "Inbetriebnahmedatum": "commissioning_date",
    "Anschlussleistung": "power_connection_[kw]",
    "Normalladeeinrichtung": "type_of_charger",
    "Anzahl Ladepunkte": "number_of_charging_points",
    "Steckertypen1": "type_of_plug_1",
    "P1 [kW]": "p1_[kw]",
    "Public Key1": "public_key1",
    "Steckertypen2": "type_of_plug_2",
    "P2 [kW]": "p2_[kw]",
    "Public Key2": "public_key2",
    "Steckertypen3": "type_of_plug_3",
    "P3 [kW]": "p3_[kw]",
    "Public Key3": "public_key3",
    "Steckertypen4": "type_of_plug_4",
    "P4 [kW]": "p4_[kw]",
    "Public Key4": "public_key4",
}

CHARGER_TYPE_MAPPING = {
    "Schnellladeeinrichtung": "fast",
    "Normalladeeinrichtung": "normal",
}

CITY_CORRECTIONS = {
    "M¸nchen": "München",
    "Frankfurt": "Frankfurt am Main",
    "Frankfurt-Niederrad": "Frankfurt am Main",
    "Stuttgart-Obertürkheim": "Stuttgart",
    "Stuttgart-Mühlhausen": "Stuttgart",
    "Stuttgart-Möhringen": "Stuttgart",
}


def load_raw_data(path: Path = RAW_CSV) -> pd.DataFrame:
    """Load the raw Ladesaeulenregister CSV with proper encoding."""
    logger.info("Loading raw data from %s", path)
    return pd.read_csv(path, encoding="latin_1", sep=";", skiprows=10)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline on the raw DataFrame.

    Returns:
        Cleaned DataFrame ready for analysis.
    """
    # 1. Rename columns to English
    df = df.rename(columns=COLUMN_MAPPING)
    logger.info("Renamed %d columns to English", len(COLUMN_MAPPING))

    # 2. Standardize charger types
    df["type_of_charger"] = df["type_of_charger"].replace(CHARGER_TYPE_MAPPING)

    # 3. Fill null values for secondary plug/power columns
    na_columns = [
        "type_of_plug_2", "p2_[kw]",
        "type_of_plug_3", "p3_[kw]",
        "type_of_plug_4", "p4_[kw]",
    ]
    for col in na_columns:
        df[col] = df[col].fillna(value="0")

    # 4. Drop public key columns (not needed for spatial analysis)
    pk_cols = ["public_key1", "public_key2", "public_key3", "public_key4"]
    df = df.drop(columns=[c for c in pk_cols if c in df.columns])

    # 5. Fix German decimal format (comma → dot) and convert to float
    numeric_cols = [
        "longitude_[dg]", "latitude_[dg]", "power_connection_[kw]",
        "p1_[kw]", "p2_[kw]", "p3_[kw]", "p4_[kw]",
    ]
    for col in numeric_cols:
        df[col] = df[col].str.replace(",", ".").astype(float)

    # 6. Parse dates (German format: DD.MM.YYYY)
    df["commissioning_date"] = pd.to_datetime(
        df["commissioning_date"], format="%d.%m.%Y"
    )

    # 7. Strip whitespace from text columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # 8. Normalize city names
    df["city"] = df["city"].replace(CITY_CORRECTIONS)

    # 9. Remove duplicates
    n_before = len(df)
    df = df.drop_duplicates()
    logger.info("Removed %d duplicate rows", n_before - len(df))

    return df


def run(input_path: Path = RAW_CSV, output_path: Path = CLEANED_CSV) -> pd.DataFrame:
    """Execute the full preprocessing pipeline and save to CSV."""
    df = load_raw_data(input_path)
    df = clean(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved cleaned dataset (%d rows) to %s", len(df), output_path)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run()
