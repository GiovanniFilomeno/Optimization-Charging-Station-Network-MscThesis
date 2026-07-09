from __future__ import annotations

import pandas as pd

from src.preprocessing import clean
from src.statistics import connector_types_by_year


def _raw_row() -> dict[str, object]:
    return {
        "Normalladeeinrichtung": "Normalladeeinrichtung",
        "Steckertypen1": "Type 2",
        "P1 [kW]": "22,0",
        "Steckertypen2": None,
        "P2 [kW]": None,
        "Steckertypen3": None,
        "P3 [kW]": None,
        "Steckertypen4": None,
        "P4 [kW]": None,
        "Breitengrad": "52,5200",
        "Lï¿½ngengrad": "13,4050",
        "Anschlussleistung": "22,0",
        "Inbetriebnahmedatum": "01.01.2022",
        "Ort": "Berlin",
    }


def test_clean_preserves_missing_connector_types_and_zero_fills_power():
    cleaned = clean(pd.DataFrame([_raw_row()]))

    assert pd.isna(cleaned.loc[0, "type_of_plug_2"])
    assert cleaned.loc[0, "p2_[kw]"] == 0.0


def test_connector_statistics_ignore_legacy_zero_sentinel():
    data = pd.DataFrame(
        {
            "year": [2021, 2022],
            "type_of_plug_1": ["Type 2", "CCS, Type 2"],
            "type_of_plug_2": ["0", None],
            "type_of_plug_3": [None, "none"],
            "type_of_plug_4": [None, ""],
        }
    )

    result = connector_types_by_year(data)

    assert set(result["connector_type"]) == {"CCS", "Type 2"}
    assert (
        result.loc[
            (result["year"] == 2022) & (result["connector_type"] == "Type 2"), "count"
        ].item()
        == 2
    )
