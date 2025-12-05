import requests
import pandas as pd

# Common USGS parameter codes
PARAMS = {
    "00010": "temperature",           # Water temp (C)
    "63680": "turbidity",             # Turbidity (often FNU)
    "00060": "flow_rate",             # Discharge (cfs) -> you can convert to m3/s
    "00095": "conductivity",          # Specific conductance (uS/cm @25C)
    "00300": "oxygen_mgL",            # Dissolved oxygen (mg/L)
    "00301": "oxygen_saturation_pct", # Dissolved oxygen (% saturation)
}

def fetch_usgs_iv(site: str,
                  period: str = "P7D",
                  parameter_cds=None,
                  timeout: int = 30) -> pd.DataFrame:
    """
    Fetch USGS Instantaneous Values (IV) data and return a tidy dataframe:
    columns: datetime + variables (where available)
    """
    if parameter_cds is None:
        parameter_cds = list(PARAMS.keys())

    url = "https://waterservices.usgs.gov/nwis/iv/"
    params = {
        "format": "json",
        "sites": site,
        "parameterCd": ",".join(parameter_cds),
        "period": period,   # e.g. P1D, P7D, P30D
        "siteStatus": "all",
    }

    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    js = r.json()

    series = js.get("value", {}).get("timeSeries", [])
    if not series:
        raise ValueError(
            "No timeSeries returned. The site may be invalid, down, or not serving the requested parameters for that period."
        )

    frames = []
    for ts in series:
        var_code = None
        vcode = ts.get("variable", {}).get("variableCode", [])
        if vcode and "value" in vcode[0]:
            var_code = vcode[0]["value"]

        # Values are nested: values[0].value is a list of points
        vals = ts.get("values", [])
        points = (vals[0].get("value", []) if vals else [])
        if not points:
            continue

        name = PARAMS.get(var_code, f"param_{var_code or 'unknown'}")

        df = pd.DataFrame(points)
        # typical keys: "value", "dateTime", "qualifiers"
        if "dateTime" not in df.columns or "value" not in df.columns:
            continue

        df = df[["dateTime", "value"]].copy()
        df["dateTime"] = pd.to_datetime(df["dateTime"], utc=True, errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["dateTime"]).rename(columns={"dateTime": "datetime", "value": name})

        frames.append(df)

    if not frames:
        raise ValueError("Returned timeSeries had no usable points.")

    # Outer-join all variables on datetime
    out = frames[0]
    for df in frames[1:]:
        out = out.merge(df, on="datetime", how="outer")

    out = out.sort_values("datetime").reset_index(drop=True)

    # Optional: convert discharge from cfs to m^3/s if you want
    # 1 cfs = 0.028316846592 m3/s
    if "flow_rate" in out.columns:
        out["flow_rate_m3s"] = out["flow_rate"] * 0.028316846592

    return out
