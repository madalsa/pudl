"""
Generate utility-level dashboard data from EIA-861 and FERC Form 1 via PUDL.

Run this script with access to PUDL nightly S3 builds to generate the
utility-level rate trends, customer counts, and sales mix data needed by
the Electricity Affordability Dashboard.

Usage:
    python generate_dashboard_data.py

Output:
    Updates dashboard_data.json with utility-level fields:
    - utility_rates: {utility_name: {residential: {years, rate, sales_twh, revenue}, ...}}
    - utility_customers: {utility_name: {residential: {years, customers, sales_twh}, ...}}
    - utility_sales_mix: {utility_name: {residential: {years, share}, ...}}
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
S3_BASE = "https://s3.us-west-2.amazonaws.com/pudl.catalyst.coop/nightly"
REAL_DOLLAR_YEAR = 2024
MIN_ANNUAL_MWH = 500_000  # minimum total sales to include utility

# CPI-U annual averages for inflation adjustment
CPI_U = {
    2001: 177.042, 2002: 179.867, 2003: 183.960, 2004: 188.908, 2005: 195.267,
    2006: 201.558, 2007: 207.344, 2008: 215.254, 2009: 214.537, 2010: 218.056,
    2011: 224.939, 2012: 229.594, 2013: 232.957, 2014: 236.736, 2015: 237.017,
    2016: 240.007, 2017: 245.120, 2018: 251.107, 2019: 255.657, 2020: 258.811,
    2021: 270.970, 2022: 292.655, 2023: 304.702, 2024: 313.689,
}
DEFLATOR = {yr: CPI_U[REAL_DOLLAR_YEAR] / cpi for yr, cpi in CPI_U.items()}

OUTPUT_JSON = Path("dashboard_data.json")
DASHBOARD_HTML = Path("rate_changes_dashboard.html")


def load_eia861_sales():
    """Load EIA-861 sales to ultimate consumers from PUDL nightly S3 build."""
    table = "core_eia861__yearly_sales_to_ultimate_consumers"
    url = f"{S3_BASE}/{table}.parquet"
    print(f"Loading {url}...")
    df = pd.read_parquet(url)
    df["year"] = pd.to_datetime(df["report_date"]).dt.year
    return df


def compute_utility_rates(sales_df):
    """Compute utility-level rate trends by customer class.

    Returns dict: {utility_name: {class: {years, rate, sales_twh, revenue}}}
    """
    # Filter to years with CPI data
    df = sales_df[sales_df["year"].isin(CPI_U.keys())].copy()

    # Standard customer classes
    class_map = {
        "residential": "residential",
        "commercial": "commercial",
        "industrial": "industrial",
    }

    # Filter to standard classes
    df = df[df["customer_class"].isin(class_map.keys())].copy()
    df["customer_class"] = df["customer_class"].map(class_map)

    # Use utility_name_eia as the display name
    name_col = "utility_name_eia"

    # Aggregate by utility, year, class
    agg = (
        df.groupby([name_col, "year", "customer_class"])
        .agg({"sales_mwh": "sum", "revenues": "sum", "customers": "sum"})
        .reset_index()
    )

    # Filter to utilities with meaningful sales in most recent year
    max_year = agg["year"].max()
    recent = agg[agg["year"] == max_year].groupby(name_col)["sales_mwh"].sum()
    big_utils = recent[recent >= MIN_ANNUAL_MWH].index

    agg = agg[agg[name_col].isin(big_utils)]

    # Compute rates in real ¢/kWh
    agg["deflator"] = agg["year"].map(DEFLATOR)
    agg["real_revenue"] = agg["revenues"] * agg["deflator"]
    agg["rate_ckwh"] = np.where(
        agg["sales_mwh"] > 0,
        (agg["real_revenue"] / agg["sales_mwh"]) * 100 / 1000,  # $/MWh -> ¢/kWh
        np.nan,
    )
    agg["sales_twh"] = agg["sales_mwh"] / 1e6
    agg["revenue_b"] = agg["real_revenue"] / 1e9

    # Build output dict
    utility_rates = {}
    for util_name, util_df in agg.groupby(name_col):
        entry = {}
        for cls in ["residential", "commercial", "industrial"]:
            cls_df = util_df[util_df["customer_class"] == cls].sort_values("year")
            if len(cls_df) < 3:
                continue
            entry[cls] = {
                "years": cls_df["year"].tolist(),
                "rate": [round(r, 1) if not np.isnan(r) else None for r in cls_df["rate_ckwh"]],
                "sales_twh": [round(s, 2) for s in cls_df["sales_twh"]],
                "revenue_b": [round(r, 3) for r in cls_df["revenue_b"]],
            }
        if entry:
            utility_rates[util_name] = entry

    return utility_rates


def compute_utility_customers(sales_df):
    """Compute utility-level customer counts and sales by class.

    Returns dict: {utility_name: {class: {years, customers, sales_twh}}}
    """
    df = sales_df[sales_df["year"].isin(CPI_U.keys())].copy()
    class_map = {"residential": "residential", "commercial": "commercial", "industrial": "industrial"}
    df = df[df["customer_class"].isin(class_map.keys())].copy()
    df["customer_class"] = df["customer_class"].map(class_map)

    name_col = "utility_name_eia"

    agg = (
        df.groupby([name_col, "year", "customer_class"])
        .agg({"sales_mwh": "sum", "customers": "sum"})
        .reset_index()
    )

    max_year = agg["year"].max()
    recent = agg[agg["year"] == max_year].groupby(name_col)["sales_mwh"].sum()
    big_utils = recent[recent >= MIN_ANNUAL_MWH].index
    agg = agg[agg[name_col].isin(big_utils)]

    agg["sales_twh"] = agg["sales_mwh"] / 1e6

    utility_customers = {}
    for util_name, util_df in agg.groupby(name_col):
        entry = {}
        for cls in ["residential", "commercial", "industrial"]:
            cls_df = util_df[util_df["customer_class"] == cls].sort_values("year")
            if len(cls_df) < 3:
                continue
            entry[cls] = {
                "years": cls_df["year"].tolist(),
                "customers": cls_df["customers"].astype(int).tolist(),
                "sales_twh": [round(s, 2) for s in cls_df["sales_twh"]],
            }
        if entry:
            utility_customers[util_name] = entry

    return utility_customers


def compute_utility_sales_mix(utility_customers):
    """Compute sales mix (% of total MWh) from customer data.

    Returns dict: {utility_name: {class: {years, share}}}
    """
    utility_mix = {}
    for util_name, classes in utility_customers.items():
        # Get all years present in all classes
        all_years = set()
        for cls_data in classes.values():
            all_years.update(cls_data["years"])
        all_years = sorted(all_years)

        # Build year -> class -> sales_twh lookup
        totals = {}
        for yr in all_years:
            total = 0
            for cls_data in classes.values():
                if yr in cls_data["years"]:
                    idx = cls_data["years"].index(yr)
                    total += cls_data["sales_twh"][idx]
            totals[yr] = total

        entry = {}
        for cls, cls_data in classes.items():
            shares = []
            years = []
            for yr, twh in zip(cls_data["years"], cls_data["sales_twh"]):
                if totals.get(yr, 0) > 0:
                    shares.append(round(100 * twh / totals[yr], 1))
                    years.append(yr)
            if shares:
                entry[cls] = {"years": years, "share": shares}

        if entry:
            utility_mix[util_name] = entry

    return utility_mix


def compute_load_growth(sales_df, base_year=2019, end_year=2024, min_twh=1.0):
    """Compute utility-level load growth for industrial AND commercial classes.

    Data centers are classified as industrial by some utilities and commercial by
    others, so both classes must be examined to identify demand hotspots.

    Returns list of dicts sorted by absolute TWh change (industrial + commercial combined).
    """
    df = sales_df[sales_df["year"].isin([base_year, end_year])].copy()
    class_map = {"commercial": "commercial", "industrial": "industrial"}
    df = df[df["customer_class"].isin(class_map.keys())].copy()
    df["customer_class"] = df["customer_class"].map(class_map)

    name_col = "utility_name_eia"

    agg = (
        df.groupby([name_col, "year", "customer_class"])
        .agg({"sales_mwh": "sum", "customers": "sum"})
        .reset_index()
    )
    agg["sales_twh"] = agg["sales_mwh"] / 1e6

    results = []
    for util_name, util_df in agg.groupby(name_col):
        entry = {"name": util_name}
        total_abs_chg = 0
        for cls in ["industrial", "commercial"]:
            cls_df = util_df[util_df["customer_class"] == cls].sort_values("year")
            base = cls_df[cls_df["year"] == base_year]
            end = cls_df[cls_df["year"] == end_year]
            if len(base) == 0 or len(end) == 0:
                continue
            twh_base = base["sales_twh"].values[0]
            twh_end = end["sales_twh"].values[0]
            cust_base = int(base["customers"].values[0])
            cust_end = int(end["customers"].values[0])
            abs_chg = twh_end - twh_base
            pct_chg = 100 * abs_chg / twh_base if twh_base > 0 else 0
            entry[f"{cls}_twh_{base_year % 100}"] = round(twh_base, 2)
            entry[f"{cls}_twh_{end_year % 100}"] = round(twh_end, 2)
            entry[f"{cls}_pct_chg"] = round(pct_chg, 1)
            entry[f"{cls}_abs_chg"] = round(abs_chg, 2)
            entry[f"{cls}_cust_{base_year % 100}"] = cust_base
            entry[f"{cls}_cust_{end_year % 100}"] = cust_end
            total_abs_chg += abs_chg

        entry["total_abs_chg"] = round(total_abs_chg, 2)
        # Only include if there's meaningful growth
        if total_abs_chg >= min_twh:
            results.append(entry)

    results.sort(key=lambda x: x["total_abs_chg"], reverse=True)
    return results


def main():
    print("=" * 60)
    print("Generating utility-level dashboard data from EIA-861")
    print("=" * 60)

    try:
        sales = load_eia861_sales()
    except Exception as e:
        print(f"\nERROR: Could not load EIA-861 data from S3: {e}")
        print("Make sure you have network access to pudl.catalyst.coop")
        print("or update S3_BASE to point to your local PUDL data.")
        sys.exit(1)

    print(f"Loaded {len(sales):,} rows, {sales['year'].min()}-{sales['year'].max()}")

    print("\nComputing utility-level rate trends...")
    utility_rates = compute_utility_rates(sales)
    print(f"  {len(utility_rates)} utilities with rate data")

    print("Computing utility-level customer data...")
    utility_customers = compute_utility_customers(sales)
    print(f"  {len(utility_customers)} utilities with customer data")

    print("Computing utility-level sales mix...")
    utility_mix = compute_utility_sales_mix(utility_customers)
    print(f"  {len(utility_mix)} utilities with mix data")

    print("Computing industrial + commercial load growth (2019-2024)...")
    load_growth = compute_load_growth(sales)
    print(f"  {len(load_growth)} utilities with significant load growth")
    if load_growth:
        print("  Top 5 by combined industrial + commercial TWh growth:")
        for item in load_growth[:5]:
            ind = item.get("industrial_abs_chg", 0)
            com = item.get("commercial_abs_chg", 0)
            print(f"    {item['name']}: ind {ind:+.1f} TWh, com {com:+.1f} TWh, total {item['total_abs_chg']:+.1f} TWh")

    # Build utility list for search (sorted by most recent total sales)
    util_list = sorted(
        utility_rates.keys(),
        key=lambda n: sum(
            utility_rates[n].get(c, {}).get("sales_twh", [0])[-1:]
            for c in ["residential", "commercial", "industrial"]
        ),
        reverse=True,
    )
    print(f"\nTop 10 utilities by sales:")
    for name in util_list[:10]:
        total = sum(
            utility_rates[name].get(c, {}).get("sales_twh", [0])[-1]
            for c in ["residential", "commercial", "industrial"]
        )
        print(f"  {name}: {total:.1f} TWh")

    # Load existing dashboard data if present
    if DASHBOARD_HTML.exists():
        import re

        with open(DASHBOARD_HTML) as f:
            html = f.read()
        match = re.search(r"const D = ({.*?});", html, re.DOTALL)
        if match:
            existing = json.loads(match.group(1))
            print(f"\nLoaded existing dashboard data with keys: {list(existing.keys())}")
        else:
            existing = {}
    else:
        existing = {}

    # Add new utility-level data
    existing["utility_rates"] = utility_rates
    existing["utility_customers"] = utility_customers
    existing["utility_mix"] = utility_mix
    existing["utility_list"] = util_list
    existing["load_growth"] = load_growth

    # Write to JSON (can be embedded into HTML later)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(existing, f, separators=(",", ":"))

    size_mb = OUTPUT_JSON.stat().st_size / 1e6
    print(f"\nWrote {OUTPUT_JSON} ({size_mb:.1f} MB)")
    print("Run the dashboard rebuild script to embed this data into the HTML.")


if __name__ == "__main__":
    main()
