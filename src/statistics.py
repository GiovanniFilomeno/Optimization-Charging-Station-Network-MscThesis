"""
Descriptive Statistics Module.

Generates comprehensive statistics about the German EV charging infrastructure:
    1. Connector counts by federal state, year, and charger type (cumulative)
    2. Power capacity analysis (total and average per state/year)
    3. Connector type trends over time
    4. Cumulative growth visualizations

Usage:
    python -m src.statistics
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import CLEANED_CSV, FIGURES_DIR, STATE_NAME_CORRECTIONS

logger = logging.getLogger(__name__)

STATS_DIR = FIGURES_DIR / "statistics"


def load_data() -> pd.DataFrame:
    """Load cleaned data and extract the commissioning year."""
    data = pd.read_csv(CLEANED_CSV)
    data["year"] = pd.to_datetime(data["commissioning_date"]).dt.year
    data["federal_state"] = data["federal_state"].replace(STATE_NAME_CORRECTIONS)
    return data


def connectors_by_state(data: pd.DataFrame) -> pd.DataFrame:
    """Cumulative connector counts per state, year, and charger type."""
    df = (
        data.groupby(["federal_state", "year", "type_of_charger"])
        .size()
        .reset_index(name="count")
    )
    df["cumulative_count"] = df.groupby(["federal_state", "type_of_charger"])["count"].cumsum()
    df["pct_change"] = (
        df.groupby(["federal_state", "type_of_charger"])["cumulative_count"].pct_change() * 100
    )
    return df


def power_by_state(data: pd.DataFrame) -> pd.DataFrame:
    """Cumulative total and average power by state and year."""
    df = (
        data.groupby(["federal_state", "year", "type_of_charger"])
        .agg({"power_connection_[kw]": ["sum", "mean"]})
        .reset_index()
    )
    df.columns = ["federal_state", "year", "type_of_charger", "total_power", "avg_power"]
    df["cumulative_total_power"] = (
        df.groupby(["federal_state", "type_of_charger"])["total_power"].cumsum()
    )
    return df


def connector_types_by_year(data: pd.DataFrame) -> pd.DataFrame:
    """Track adoption of different plug standards over time."""
    cols = ["type_of_plug_1", "type_of_plug_2", "type_of_plug_3", "type_of_plug_4"]
    melted = pd.melt(
        data, id_vars=["year"], value_vars=cols,
        var_name="connector_column", value_name="connector_type",
    ).dropna()
    melted["connector_type"] = melted["connector_type"].apply(lambda x: x.split(", "))
    melted = melted.explode("connector_type")
    df = (
        melted.groupby(["year", "connector_type"]).size()
        .groupby("connector_type").cumsum()
        .reset_index(name="count")
    )
    df["pct_change"] = df.groupby("connector_type")["count"].pct_change() * 100
    return df


def plot_cumulative_growth(data: pd.DataFrame) -> None:
    """Bar chart: cumulative normal vs. fast charging stations."""
    pivot = data.pivot_table(
        index="year", columns="type_of_charger",
        values="power_connection_[kw]", aggfunc="count",
    ).fillna(0)
    cum = pivot.cumsum().reindex(range(2009, pivot.index.max() + 1)).fillna(method="ffill")

    plt.rcParams["font.size"] = 14
    fig, ax = plt.subplots(figsize=(20, 6))
    cum.plot(kind="bar", stacked=False, ax=ax)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Stations")
    ax.legend(title="Type of Charger")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + 0.02, p.get_height() + 150))
    fig.savefig(STATS_DIR / "cumulative_normal_vs_fast.png", bbox_inches="tight", dpi=150)
    plt.show()


def plot_total_growth(data: pd.DataFrame) -> None:
    """Bar chart: total cumulative charging stations in Germany."""
    total = data["year"].value_counts().sort_index().cumsum()
    total = total.reindex(range(2009, total.index.max() + 1)).fillna(method="ffill")

    plt.figure(figsize=(12, 6))
    ax = total.plot(kind="bar")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Stations")
    for p in ax.patches:
        ax.annotate(str(int(p.get_height())), (p.get_x() + 0.02, p.get_height() + 150))
    plt.savefig(STATS_DIR / "total_cumulative_stations.png", bbox_inches="tight", dpi=150)
    plt.show()


def plot_power_distribution(data: pd.DataFrame) -> None:
    """Stacked bar: power range distribution over time."""
    bins = [0, 3.7, 15, 22, 49, 59, 149, 299, np.inf]
    labels = ["0–3.7", "3.7–15", "15–22", "22–49", "49–59", "59–149", "149–299", ">299"]
    data = data.copy()
    data["power_range"] = pd.cut(data["power_connection_[kw]"], bins=bins, labels=labels)
    pr = data.groupby(["year", "power_range"]).size().unstack().fillna(0).cumsum()
    pr = pr.reindex(range(2009, pr.index.max() + 1)).fillna(method="ffill")

    ax = pr.plot(kind="bar", stacked=True)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Stations")
    ax.legend(title="Power Range (kW)")
    plt.savefig(STATS_DIR / "power_range_distribution.png", bbox_inches="tight", dpi=150)
    plt.show()


def run() -> None:
    """Execute the full statistics pipeline."""
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()

    # Export tabular statistics
    connectors_by_state(data).to_excel(STATS_DIR / "connectors_by_state.xlsx", index=False)
    power_by_state(data).to_excel(STATS_DIR / "power_by_state.xlsx", index=False)
    connector_types_by_year(data).to_excel(STATS_DIR / "connector_types.xlsx", index=False)
    logger.info("Exported Excel files to %s", STATS_DIR)

    # Generate plots
    plot_cumulative_growth(data)
    plot_total_growth(data)
    plot_power_distribution(data)
    logger.info("All visualizations saved")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run()
