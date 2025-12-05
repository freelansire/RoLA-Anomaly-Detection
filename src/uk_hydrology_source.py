import requests
import pandas as pd

BASE = "https://environment.data.gov.uk/hydrology"

# ObservedProperty filter values supported by the API docs
OBSERVED_PROPERTIES = [
    "waterFlow", "waterLevel", "rainfall", "groundwaterLevel",
    "dissolved-oxygen", "fdom", "bga", "turbidity", "chlorophyll",
    "conductivity", "temperature", "ammonium", "nitrate", "ph"
]

def _safe_label(x):
    if x is None:
        return None
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return x.get("label") or x.get("value") or str(x)
    if isinstance(x, list):
        for item in x:
            if isinstance(item, dict) and ("label" in item or "value" in item):
                return item.get("label") or item.get("value")
        return str(x[0]) if len(x) else None
    return str(x)

def search_stations(search: str = "", observed_property: str | None = None,
                    status: str | None = "Active", limit: int = 60, timeout: int = 30) -> pd.DataFrame:
    url = f"{BASE}/id/stations"
    params = {"_limit": int(limit)}
    if search:
        params["search"] = search
    if status:
        params["status.label"] = status
    if observed_property:
        params["observedProperty"] = observed_property

    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    js = r.json()
    items = js.get("items", js) if isinstance(js, dict) else js
    if not isinstance(items, list):
        items = []

    rows = []
    for it in items:
        if not isinstance(it, dict):
            continue
        station_guid = it.get("stationGuid") or it.get("notation") or it.get("@id")
        rows.append({
            "stationGuid": str(station_guid) if station_guid is not None else None,
            "label": it.get("label"),
            "notation": it.get("notation"),
            "lat": it.get("lat"),
            "long": it.get("long"),
            "status": _safe_label(it.get("status")),
        })

    df = pd.DataFrame(rows).dropna(subset=["stationGuid"])
    if len(df) == 0:
        return df
    df["stationGuid"] = df["stationGuid"].astype(str)
    return df.drop_duplicates(subset=["stationGuid"]).reset_index(drop=True)

def list_station_measures(station: str,
                          observed_property: str | None = None,
                          observation_type: str = "Qualified",
                          period_name: str | None = "15min",
                          timeout: int = 30) -> pd.DataFrame:
    """
    Uses: /hydrology/id/stations/{station}/measures  (docs)
    Allows filters (observedProperty, observationType, etc.). :contentReference[oaicite:1]{index=1}
    """
    station_id = station.split("/")[-1]  # if a URL was passed in
    url = f"{BASE}/id/stations/{station_id}/measures"
    params = {"_limit": 200}
    if observed_property:
        params["observedProperty"] = observed_property
    if observation_type:
        params["observationType"] = observation_type
    if period_name:
        params["periodName"] = period_name

    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    js = r.json()
    items = js.get("items", js) if isinstance(js, dict) else js
    if not isinstance(items, list):
        items = []

    rows = []
    for it in items:
        if not isinstance(it, dict):
            continue
        rows.append({
            "id": (it.get("@id") or it.get("notation") or ""),
            "notation": it.get("notation"),
            "label": it.get("label"),
            "parameter": it.get("parameter"),
            "parameterName": it.get("parameterName"),
            "observedProperty": _safe_label(it.get("observedProperty")),
            "observedProperty.label": it.get("observedProperty.label") or _safe_label((it.get("observedProperty") or {}).get("label") if isinstance(it.get("observedProperty"), dict) else None),
            "unitName": it.get("unitName"),
            "periodName": it.get("periodName"),
            "valueType": it.get("valueType"),
            "observationType.label": it.get("observationType.label") or _safe_label(it.get("observationType")),
        })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    # keep only measures with ids
    df = df[df["id"].astype(str).str.len() > 0].copy()
    df["measure"] = df["id"].astype(str).apply(lambda s: s.split("/measures/")[-1] if "/measures/" in s else s.split("/")[-1])
    # Sort “nicely”
    return df.sort_values(["periodName", "parameterName", "unitName"], na_position="last").reset_index(drop=True)

def _measure_to_column(parameter_name: str | None, unit_name: str | None) -> str:
    p = (parameter_name or "").strip().lower()
    u = (unit_name or "").strip().lower()

    if "flow" in p:
        return "flow_rate"
    if "level" in p:
        return "water_level"
    if "temperature" in p:
        return "temperature"
    if "turbidity" in p:
        return "turbidity"
    if "conductivity" in p:
        return "conductivity"
    if "dissolved oxygen" in p or "oxygen" in p:
        if "%" in u:
            return "oxygen_saturation_pct"
        if "mg" in u:
            return "oxygen_mgL"
        return "oxygen"

    # fallback: safe column name
    base = p.replace(" ", "_").replace("-", "_") or "signal"
    return base

def fetch_measure_readings(measure: str, min_date: str, max_date: str,
                           limit: int = 100000, timeout: int = 30) -> pd.DataFrame:
    """
    Uses /hydrology/id/measures/{measure}/readings with mineq-date/max-date. :contentReference[oaicite:2]{index=2}
    """
    measure_id = measure.split("/")[-1]
    url = f"{BASE}/id/measures/{measure_id}/readings"
    params = {"mineq-date": min_date, "max-date": max_date, "_limit": int(limit)}

    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    js = r.json()
    items = js.get("items", js) if isinstance(js, dict) else js
    if not isinstance(items, list) or not items:
        return pd.DataFrame(columns=["datetime", "value"])

    df = pd.DataFrame(items)
    # API uses dateTime/value keys
    if "dateTime" in df.columns:
        df = df.rename(columns={"dateTime": "datetime"})
    if "value" not in df.columns:
        df["value"] = pd.NA

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return df[["datetime", "value"]]

def fetch_measures_bundle(measures_df: pd.DataFrame, selected_measures: list[str],
                          min_date: str, max_date: str, timeout: int = 30) -> pd.DataFrame:
    """
    Fetch readings for selected measures and merge on datetime.
    """
    out = None
    for mid in selected_measures:
        row = measures_df.loc[measures_df["measure"] == mid]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        col = _measure_to_column(row.get("parameterName"), row.get("unitName"))

        df = fetch_measure_readings(mid, min_date, max_date, timeout=timeout)
        if df.empty:
            continue
        df = df.rename(columns={"value": col})

        out = df if out is None else out.merge(df, on="datetime", how="outer")

    if out is None:
        return pd.DataFrame(columns=["datetime"])
    return out.sort_values("datetime").reset_index(drop=True)
