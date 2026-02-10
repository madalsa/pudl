"""
Regional Electricity Cost Analysis: High-Cost Northeast & Appalachian IOUs

Analysis of revenue requirements, O&M costs, rate base, and financial metrics for
Investor-Owned Utilities (IOUs) in high electricity cost states:
Maine, New York, Massachusetts, West Virginia, and Maryland.

Data source: PUDL FERC Form 1 (2015-2024), accessed via the nightly S3 parquet builds.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
S3_BASE = "https://s3.us-west-2.amazonaws.com/pudl.catalyst.coop/nightly"
YEARS = range(2015, 2025)
BASE_YEAR = 2015
OUTPUT_DIR = Path("output_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# Major IOUs in each target state (utility_id_ferc1 -> metadata)
TARGET_UTILITIES = {
    # Maine
    212: {"name": "Central Maine Power", "state": "ME", "color": "#1f77b4"},
    158: {"name": "Emera Maine (Versant)", "state": "ME", "color": "#aec7e8"},
    # New York
    159: {"name": "Consolidated Edison", "state": "NY", "color": "#ff7f0e"},
    246: {"name": "Central Hudson G&E", "state": "NY", "color": "#ffbb78"},
    215: {"name": "NY State Electric & Gas", "state": "NY", "color": "#d62728"},
    275: {"name": "Niagara Mohawk (National Grid)", "state": "NY", "color": "#e377c2"},
    214: {"name": "Rochester Gas & Electric", "state": "NY", "color": "#ff9896"},
    190: {"name": "Orange & Rockland Utilities", "state": "NY", "color": "#c49c94"},
    # Massachusetts
    277: {"name": "Massachusetts Electric (National Grid)", "state": "MA", "color": "#2ca02c"},
    271: {"name": "NSTAR Electric (Eversource)", "state": "MA", "color": "#98df8a"},
    245: {"name": "Western Mass Electric (Eversource)", "state": "MA", "color": "#7f7f7f"},
    # West Virginia
    238: {"name": "Monongahela Power", "state": "WV", "color": "#9467bd"},
    200: {"name": "Appalachian Power", "state": "WV", "color": "#c5b0d5"},
    240: {"name": "Potomac Edison", "state": "WV", "color": "#8c564b"},
    # Maryland
    248: {"name": "Baltimore Gas & Electric", "state": "MD", "color": "#bcbd22"},
    290: {"name": "Potomac Electric Power (Pepco)", "state": "MD", "color": "#dbdb8d"},
    291: {"name": "Delmarva Power & Light", "state": "MD", "color": "#17becf"},
}

UTIL_IDS = list(TARGET_UTILITIES.keys())
UTIL_NAMES = {uid: info["name"] for uid, info in TARGET_UTILITIES.items()}
UTIL_COLORS = {uid: info["color"] for uid, info in TARGET_UTILITIES.items()}
UTIL_STATES = {uid: info["state"] for uid, info in TARGET_UTILITIES.items()}

STATE_UTILS = {}
for uid, info in TARGET_UTILITIES.items():
    STATE_UTILS.setdefault(info["state"], []).append(uid)

STATE_COLORS = {
    "ME": "#1f77b4", "NY": "#ff7f0e", "MA": "#2ca02c",
    "WV": "#9467bd", "MD": "#bcbd22",
}
STATE_NAMES = {
    "ME": "Maine", "NY": "New York", "MA": "Massachusetts",
    "WV": "West Virginia", "MD": "Maryland",
}

# Plotting defaults
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["xtick.labelsize"] = 13
plt.rcParams["ytick.labelsize"] = 13
plt.rcParams["legend.fontsize"] = 11
plt.rcParams["figure.titlesize"] = 18
plt.rcParams["axes.xmargin"] = 0.02

# ---------------------------------------------------------------------------
# 1. Load FERC Form 1 Data from PUDL
# ---------------------------------------------------------------------------
print("Loading FERC Form 1 data from PUDL nightly S3 builds...")
print(f"Analyzing {len(TARGET_UTILITIES)} IOUs across {len(STATE_UTILS)} states")
for st, uids in STATE_UTILS.items():
    print(f"  {STATE_NAMES[st]}: {', '.join(UTIL_NAMES[u] for u in uids)}")


def load_and_filter(table_name):
    """Load a parquet table from S3 and filter to target utilities and years."""
    df = pd.read_parquet(f"{S3_BASE}/{table_name}.parquet")
    return df[
        (df["utility_id_ferc1"].isin(UTIL_IDS))
        & (df["report_year"].isin(YEARS))
    ].copy()


print("\nLoading tables...")
opex = load_and_filter("core_ferc1__yearly_operating_expenses_sched320")
print(f"  Operating Expenses (sched 320): {len(opex)} rows")

plant = load_and_filter("core_ferc1__yearly_plant_in_service_sched204")
print(f"  Plant in Service (sched 204): {len(plant)} rows")

rev = load_and_filter("core_ferc1__yearly_operating_revenues_sched300")
print(f"  Operating Revenues (sched 300): {len(rev)} rows")

inc = load_and_filter("core_ferc1__yearly_income_statements_sched114")
print(f"  Income Statements (sched 114): {len(inc)} rows")

dep = load_and_filter("core_ferc1__yearly_depreciation_summary_sched336")
print(f"  Depreciation Summary (sched 336): {len(dep)} rows")


# ---------------------------------------------------------------------------
# 2. O&M Expenses by Functional Category
# ---------------------------------------------------------------------------
print("\n--- O&M Expenses Analysis ---")

OPEX_CATEGORIES = {
    "generation_expenses": "Generation",
    "transmission_expenses": "Transmission",
    "distribution_expenses": "Distribution",
    "customer_account_expenses": "Customer Accounts",
    "customer_service_and_information_expenses": "Customer Service",
    "sales_expenses": "Sales",
    "administrative_and_general_expenses": "Admin & General",
}

opex_cat = opex[opex["expense_type"].isin(OPEX_CATEGORIES.keys())].copy()
opex_cat["category"] = opex_cat["expense_type"].map(OPEX_CATEGORIES)
opex_cat["utility_name"] = opex_cat["utility_id_ferc1"].map(UTIL_NAMES)
opex_cat["state"] = opex_cat["utility_id_ferc1"].map(UTIL_STATES)

opex_pivot = opex_cat.pivot_table(
    index=["utility_id_ferc1", "utility_name", "state", "report_year"],
    columns="category",
    values="dollar_value",
    aggfunc="sum",
).reset_index()

# --- Figure 1: O&M by functional category (absolute) ---
fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharey=False)
for ax, cat in zip(axes, ["Generation", "Transmission", "Distribution"]):
    for uid in UTIL_IDS:
        df_u = opex_pivot[opex_pivot["utility_id_ferc1"] == uid].sort_values(
            "report_year"
        )
        if cat in df_u.columns and df_u[cat].notna().any():
            ax.plot(
                df_u["report_year"],
                df_u[cat] / 1e9,
                color=UTIL_COLORS[uid],
                lw=2.5,
                label=UTIL_NAMES[uid],
            )
    ax.set_title(f"{cat} O&M Expenses", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Billion $")
    ax.grid(lw=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

axes[0].legend(bbox_to_anchor=(0, -0.25), loc="upper left", ncol=3, fontsize=9)
plt.suptitle(
    "O&M Expenses by Functional Category (Nominal $)\nFERC Form 1, 2015-2024",
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_opex_by_category.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Figure 2: O&M normalized (base year = 1.0) ---
fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharey=True)
for ax, cat in zip(axes, ["Generation", "Transmission", "Distribution"]):
    for uid in UTIL_IDS:
        df_u = opex_pivot[opex_pivot["utility_id_ferc1"] == uid].sort_values(
            "report_year"
        )
        if cat not in df_u.columns or df_u[cat].isna().all():
            continue
        base_val = df_u[df_u["report_year"] == BASE_YEAR][cat]
        if base_val.empty or base_val.item() == 0:
            continue
        normalized = df_u[cat] / base_val.item()
        ax.plot(
            df_u["report_year"],
            normalized,
            color=UTIL_COLORS[uid],
            lw=2.5,
            label=UTIL_NAMES[uid],
        )
    ax.axhline(y=1, linestyle="dashed", color="grey", lw=1)
    ax.set_title(f"{cat} O&M (Normalized)", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Relative to {BASE_YEAR}")
    ax.grid(lw=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

axes[0].legend(bbox_to_anchor=(0, -0.25), loc="upper left", ncol=3, fontsize=9)
plt.suptitle(
    f"O&M Expenses Normalized ({BASE_YEAR} = 1.0)\nFERC Form 1",
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_opex_normalized.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Figure 3: State-aggregated O&M ---
state_opex = (
    opex_pivot.groupby(["state", "report_year"])[
        ["Generation", "Transmission", "Distribution", "Admin & General"]
    ]
    .sum()
    .reset_index()
)

fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharey=False)
for ax, cat in zip(axes, ["Generation", "Transmission", "Distribution"]):
    for st in STATE_COLORS:
        df_s = state_opex[state_opex["state"] == st].sort_values("report_year")
        if cat in df_s.columns and df_s[cat].notna().any():
            ax.plot(
                df_s["report_year"],
                df_s[cat] / 1e9,
                color=STATE_COLORS[st],
                lw=3,
                label=STATE_NAMES[st],
            )
    ax.set_title(f"{cat} O&M by State", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Billion $")
    ax.grid(lw=0.3)
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

plt.suptitle(
    "State-Aggregated O&M Expenses (Nominal $)\nFERC Form 1, 2015-2024",
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_opex_by_state.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Figure 4: O&M stacked bar for latest year ---
latest_year = opex_pivot["report_year"].max()
latest = opex_pivot[opex_pivot["report_year"] == latest_year].copy()
latest = latest.set_index("utility_name")

stack_cols = [
    "Generation",
    "Transmission",
    "Distribution",
    "Admin & General",
    "Customer Accounts",
    "Customer Service",
    "Sales",
]
stack_cols = [c for c in stack_cols if c in latest.columns]

fig, ax = plt.subplots(figsize=(16, 8))
latest[stack_cols].div(1e9).plot(
    kind="barh", stacked=True, ax=ax, colormap="tab10", edgecolor="white"
)
ax.set_xlabel("Billion $")
ax.set_title(
    f"O&M Cost Breakdown by Utility ({latest_year})\nFERC Form 1 Schedule 320",
    fontweight="bold",
)
ax.legend(loc="lower right")
ax.grid(axis="x", lw=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_opex_stacked_bar.png", dpi=150, bbox_inches="tight")
plt.show()


# ---------------------------------------------------------------------------
# 3. Rate Base / Plant in Service
# ---------------------------------------------------------------------------
print("\n--- Rate Base Analysis ---")

PLANT_CATEGORIES = {
    "production_plant": "Production",
    "steam_production_plant": "Steam Production",
    "nuclear_production_plant": "Nuclear Production",
    "hydraulic_production_plant": "Hydro Production",
    "other_production_plant": "Other Production",
    "transmission_plant": "Transmission",
    "distribution_plant": "Distribution",
    "general_plant": "General",
    "intangible_plant": "Intangible",
    "electric_plant_in_service": "Total Electric Plant",
}

plant_cat = plant[
    (plant["ferc_account_label"].isin(PLANT_CATEGORIES.keys()))
    & (plant["plant_status"] == "in_service")
].copy()
plant_cat["category"] = plant_cat["ferc_account_label"].map(PLANT_CATEGORIES)
plant_cat["utility_name"] = plant_cat["utility_id_ferc1"].map(UTIL_NAMES)
plant_cat["state"] = plant_cat["utility_id_ferc1"].map(UTIL_STATES)

plant_pivot = plant_cat.pivot_table(
    index=["utility_id_ferc1", "utility_name", "state", "report_year"],
    columns="category",
    values="ending_balance",
    aggfunc="sum",
).reset_index()

# --- Figure 5: Plant in service (absolute) ---
fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharey=False)
for ax, cat in zip(axes, ["Production", "Transmission", "Distribution"]):
    for uid in UTIL_IDS:
        df_u = plant_pivot[plant_pivot["utility_id_ferc1"] == uid].sort_values(
            "report_year"
        )
        if cat in df_u.columns and df_u[cat].notna().any():
            ax.plot(
                df_u["report_year"],
                df_u[cat] / 1e9,
                color=UTIL_COLORS[uid],
                lw=2.5,
                label=UTIL_NAMES[uid],
            )
    ax.set_title(f"{cat} Plant in Service", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Billion $")
    ax.grid(lw=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

axes[0].legend(bbox_to_anchor=(0, -0.25), loc="upper left", ncol=3, fontsize=9)
plt.suptitle(
    "Electric Plant in Service (Nominal $)\nFERC Form 1 Schedule 204, 2015-2024",
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_plant_in_service.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Figure 6: Plant in service normalized ---
fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharey=True)
for ax, cat in zip(axes, ["Production", "Transmission", "Distribution"]):
    for uid in UTIL_IDS:
        df_u = plant_pivot[plant_pivot["utility_id_ferc1"] == uid].sort_values(
            "report_year"
        )
        if cat not in df_u.columns or df_u[cat].isna().all():
            continue
        base_val = df_u[df_u["report_year"] == BASE_YEAR][cat]
        if base_val.empty or base_val.item() == 0:
            continue
        normalized = df_u[cat] / base_val.item()
        ax.plot(
            df_u["report_year"],
            normalized,
            color=UTIL_COLORS[uid],
            lw=2.5,
            label=UTIL_NAMES[uid],
        )
    ax.axhline(y=1, linestyle="dashed", color="grey", lw=1)
    ax.set_title(f"{cat} Plant (Normalized)", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Relative to {BASE_YEAR}")
    ax.grid(lw=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

axes[0].legend(bbox_to_anchor=(0, -0.25), loc="upper left", ncol=3, fontsize=9)
plt.suptitle(
    f"Plant in Service Normalized ({BASE_YEAR} = 1.0)\nFERC Form 1 Schedule 204",
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "06_plant_in_service_normalized.png", dpi=150, bbox_inches="tight"
)
plt.show()

# --- Figure 7: Capital Additions ---
add_pivot = plant_cat.pivot_table(
    index=["utility_id_ferc1", "utility_name", "state", "report_year"],
    columns="category",
    values="additions",
    aggfunc="sum",
).reset_index()

fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharey=False)
for ax, cat in zip(axes, ["Production", "Transmission", "Distribution"]):
    for uid in UTIL_IDS:
        df_u = add_pivot[add_pivot["utility_id_ferc1"] == uid].sort_values(
            "report_year"
        )
        if cat in df_u.columns and df_u[cat].notna().any():
            ax.plot(
                df_u["report_year"],
                df_u[cat] / 1e9,
                color=UTIL_COLORS[uid],
                lw=2.5,
                label=UTIL_NAMES[uid],
            )
    ax.set_title(f"{cat} Capital Additions", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Billion $")
    ax.grid(lw=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

axes[0].legend(bbox_to_anchor=(0, -0.25), loc="upper left", ncol=3, fontsize=9)
plt.suptitle(
    "Annual Capital Additions (Nominal $)\nFERC Form 1 Schedule 204, 2015-2024",
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "07_capital_additions.png", dpi=150, bbox_inches="tight")
plt.show()


# ---------------------------------------------------------------------------
# 4. Revenue & Electricity Sales
# ---------------------------------------------------------------------------
print("\n--- Revenue & Sales Analysis ---")

# Total operating revenues
total_rev = rev[rev["revenue_type"] == "electric_operating_revenues"].copy()
total_rev["utility_name"] = total_rev["utility_id_ferc1"].map(UTIL_NAMES)
total_rev["state"] = total_rev["utility_id_ferc1"].map(UTIL_STATES)

# Revenue by customer class
REV_CLASSES = {
    "residential_sales": "Residential",
    "small_or_commercial": "Commercial",
    "large_or_industrial": "Industrial",
    "sales_to_ultimate_consumers": "Total Retail",
    "sales_for_resale": "Wholesale",
}

rev_class = rev[rev["revenue_type"].isin(REV_CLASSES.keys())].copy()
rev_class["customer_class"] = rev_class["revenue_type"].map(REV_CLASSES)
rev_class["utility_name"] = rev_class["utility_id_ferc1"].map(UTIL_NAMES)
rev_class["state"] = rev_class["utility_id_ferc1"].map(UTIL_STATES)

# --- Figure 8: Revenue and Retail Sales ---
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

ax = axes[0]
for uid in UTIL_IDS:
    df_u = total_rev[total_rev["utility_id_ferc1"] == uid].sort_values("report_year")
    if len(df_u) > 0:
        ax.plot(
            df_u["report_year"],
            df_u["dollar_value"] / 1e9,
            color=UTIL_COLORS[uid],
            lw=2.5,
            label=UTIL_NAMES[uid],
        )
ax.set_title("Total Electric Operating Revenues", fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Billion $")
ax.grid(lw=0.3)
ax.legend(fontsize=8, ncol=2)
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

ax = axes[1]
retail = rev_class[rev_class["customer_class"] == "Total Retail"]
for uid in UTIL_IDS:
    df_u = retail[retail["utility_id_ferc1"] == uid].sort_values("report_year")
    if len(df_u) > 0 and df_u["sales_mwh"].notna().any():
        ax.plot(
            df_u["report_year"],
            df_u["sales_mwh"] / 1e6,
            color=UTIL_COLORS[uid],
            lw=2.5,
            label=UTIL_NAMES[uid],
        )
ax.set_title("Retail Electricity Sales", fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Million MWh (TWh)")
ax.grid(lw=0.3)
ax.legend(fontsize=8, ncol=2)
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

plt.suptitle(
    "Revenue & Sales\nFERC Form 1, 2015-2024", fontweight="bold", y=1.02
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "08_revenue_and_sales.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Figure 9: Average Electricity Prices ---
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

for ax, cls in zip(axes, ["Residential", "Total Retail"]):
    cls_data = rev_class[rev_class["customer_class"] == cls]
    for uid in UTIL_IDS:
        df_u = cls_data[cls_data["utility_id_ferc1"] == uid].sort_values("report_year")
        if (
            len(df_u) > 0
            and df_u["sales_mwh"].notna().any()
            and (df_u["sales_mwh"] > 0).any()
        ):
            avg_price = df_u["dollar_value"] / df_u["sales_mwh"]  # $/MWh
            ax.plot(
                df_u["report_year"],
                avg_price * 0.1,  # convert to cents/kWh
                color=UTIL_COLORS[uid],
                lw=2.5,
                label=UTIL_NAMES[uid],
            )
    ax.set_title(f"Average {cls} Electricity Price", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("cents/kWh")
    ax.grid(lw=0.3)
    ax.legend(fontsize=8, ncol=2)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

plt.suptitle(
    "Average Electricity Prices\nFERC Form 1 Schedule 300, 2015-2024",
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "09_avg_electricity_prices.png", dpi=150, bbox_inches="tight")
plt.show()


# ---------------------------------------------------------------------------
# 5. Income Statement Analysis
# ---------------------------------------------------------------------------
print("\n--- Income Statement Analysis ---")

INCOME_ITEMS = {
    "operating_revenues": "Operating Revenues",
    "utility_operating_expenses": "Operating Expenses",
    "net_utility_operating_income": "Net Operating Income",
    "depreciation_expense": "Depreciation",
    "income_taxes_operating_income": "Operating Income Taxes",
    "maintenance_expense": "Maintenance",
    "operation_expense": "Operations",
    "net_income_loss": "Net Income",
}

inc_items = inc[
    (inc["income_type"].isin(INCOME_ITEMS.keys())) & (inc["utility_type"] == "electric")
].copy()
inc_items["item"] = inc_items["income_type"].map(INCOME_ITEMS)
inc_items["utility_name"] = inc_items["utility_id_ferc1"].map(UTIL_NAMES)
inc_items["state"] = inc_items["utility_id_ferc1"].map(UTIL_STATES)

inc_pivot = inc_items.pivot_table(
    index=["utility_id_ferc1", "utility_name", "state", "report_year"],
    columns="item",
    values="dollar_value",
    aggfunc="sum",
).reset_index()

# --- Figure 10: Income statement highlights ---
fig, axes = plt.subplots(1, 3, figsize=(24, 8))
for ax, item in zip(
    axes, ["Operating Revenues", "Operating Expenses", "Net Operating Income"]
):
    for uid in UTIL_IDS:
        df_u = inc_pivot[inc_pivot["utility_id_ferc1"] == uid].sort_values(
            "report_year"
        )
        if item in df_u.columns and df_u[item].notna().any():
            ax.plot(
                df_u["report_year"],
                df_u[item] / 1e9,
                color=UTIL_COLORS[uid],
                lw=2.5,
                label=UTIL_NAMES[uid],
            )
    ax.set_title(item, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Billion $")
    ax.grid(lw=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

axes[0].legend(bbox_to_anchor=(0, -0.25), loc="upper left", ncol=3, fontsize=9)
plt.suptitle(
    "Income Statement Highlights\nFERC Form 1 Schedule 114, 2015-2024",
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "10_income_statement_highlights.png", dpi=150, bbox_inches="tight"
)
plt.show()


# --- Figure 11: Implied Return on Rate Base ---
noi = inc_pivot[["utility_id_ferc1", "report_year", "Net Operating Income"]].dropna()
total_plant_series = plant_pivot[plant_pivot["Total Electric Plant"].notna()][
    ["utility_id_ferc1", "report_year", "Total Electric Plant"]
]

ror_df = noi.merge(total_plant_series, on=["utility_id_ferc1", "report_year"])
ror_df["implied_ror"] = (
    ror_df["Net Operating Income"] / ror_df["Total Electric Plant"] * 100
)
ror_df["utility_name"] = ror_df["utility_id_ferc1"].map(UTIL_NAMES)

fig, ax = plt.subplots(figsize=(14, 8))
for uid in UTIL_IDS:
    df_u = ror_df[ror_df["utility_id_ferc1"] == uid].sort_values("report_year")
    if len(df_u) > 0:
        ax.plot(
            df_u["report_year"],
            df_u["implied_ror"],
            color=UTIL_COLORS[uid],
            lw=2.5,
            marker="o",
            markersize=4,
            label=UTIL_NAMES[uid],
        )
ax.set_title(
    "Implied Return on Rate Base\n(Net Operating Income / Total Electric Plant)",
    fontweight="bold",
)
ax.set_xlabel("Year")
ax.set_ylabel("%")
ax.grid(lw=0.3)
ax.legend(fontsize=9, ncol=3, loc="upper right")
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "11_implied_return_on_rate_base.png", dpi=150, bbox_inches="tight")
plt.show()


# ---------------------------------------------------------------------------
# 6. Revenue Requirement Decomposition
# ---------------------------------------------------------------------------
print("\n--- Revenue Requirement Decomposition ---")

# Pick the biggest utility per state for clarity
largest_by_state = {}
for st, uids in STATE_UTILS.items():
    rev_by_uid = total_rev.groupby("utility_id_ferc1")["dollar_value"].sum()
    best_uid = max(uids, key=lambda u: rev_by_uid.get(u, 0))
    largest_by_state[st] = best_uid

# --- Figure 12: Revenue requirement decomposition ---
rr_components = inc_pivot[inc_pivot["report_year"].isin(YEARS)]
fig, axes = plt.subplots(
    len(largest_by_state), 1, figsize=(14, 5 * len(largest_by_state)), sharex=True
)

for ax, (st, uid) in zip(axes, largest_by_state.items()):
    df_u = rr_components[rr_components["utility_id_ferc1"] == uid].sort_values(
        "report_year"
    )
    if len(df_u) == 0:
        continue

    components = []
    labels = []
    colors = []
    comp_map = [
        ("Operations", "tab:blue"),
        ("Maintenance", "tab:cyan"),
        ("Depreciation", "tab:orange"),
        ("Operating Income Taxes", "tab:red"),
        ("Net Operating Income", "tab:green"),
    ]
    for comp_name, comp_color in comp_map:
        if comp_name in df_u.columns and df_u[comp_name].notna().any():
            components.append(df_u[comp_name].fillna(0).values / 1e9)
            labels.append(comp_name)
            colors.append(comp_color)

    if components:
        ax.stackplot(
            df_u["report_year"].values,
            *components,
            labels=labels,
            colors=colors,
            alpha=0.8,
        )
        if "Operating Revenues" in df_u.columns:
            ax.plot(
                df_u["report_year"],
                df_u["Operating Revenues"] / 1e9,
                color="black",
                lw=3,
                linestyle="--",
                label="Total Revenue",
            )

    ax.set_title(f"{UTIL_NAMES[uid]} ({STATE_NAMES[st]})", fontweight="bold")
    ax.set_ylabel("Billion $")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(lw=0.3)

axes[-1].set_xlabel("Year")
plt.suptitle(
    "Revenue Requirement Decomposition\nFERC Form 1 Schedule 114",
    fontweight="bold",
    y=1.01,
)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "12_revenue_requirement_decomposition.png",
    dpi=150,
    bbox_inches="tight",
)
plt.show()


# ---------------------------------------------------------------------------
# 7. O&M Cost per MWh
# ---------------------------------------------------------------------------
print("\n--- O&M Cost per MWh ---")

retail_sales = rev_class[rev_class["customer_class"] == "Total Retail"][
    ["utility_id_ferc1", "report_year", "sales_mwh"]
].rename(columns={"sales_mwh": "retail_mwh"})

cost_per_mwh = opex_pivot.merge(
    retail_sales, on=["utility_id_ferc1", "report_year"]
)

# --- Figure 13: O&M cost per MWh ---
fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharey=False)
for ax, cat in zip(axes, ["Generation", "Transmission", "Distribution"]):
    for uid in UTIL_IDS:
        df_u = cost_per_mwh[cost_per_mwh["utility_id_ferc1"] == uid].sort_values(
            "report_year"
        )
        if (
            cat in df_u.columns
            and df_u[cat].notna().any()
            and (df_u["retail_mwh"] > 0).any()
        ):
            cost_intensity = df_u[cat] / df_u["retail_mwh"]  # $/MWh
            ax.plot(
                df_u["report_year"],
                cost_intensity,
                color=UTIL_COLORS[uid],
                lw=2.5,
                label=UTIL_NAMES[uid],
            )
    ax.set_title(f"{cat} O&M per MWh", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("$/MWh")
    ax.grid(lw=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

axes[0].legend(bbox_to_anchor=(0, -0.25), loc="upper left", ncol=3, fontsize=9)
plt.suptitle(
    "O&M Cost Intensity ($/MWh Retail Sales)\nFERC Form 1, 2015-2024",
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "13_opex_per_mwh.png", dpi=150, bbox_inches="tight")
plt.show()


# ---------------------------------------------------------------------------
# 8. Cross-State Comparison
# ---------------------------------------------------------------------------
print("\n--- Cross-State Comparison ---")

# Average price by state
rev_class_state = rev_class.copy()
state_retail = (
    rev_class_state[rev_class_state["customer_class"] == "Total Retail"]
    .groupby(["state", "report_year"])
    .agg(total_revenue=("dollar_value", "sum"), total_mwh=("sales_mwh", "sum"))
    .reset_index()
)
state_retail["avg_price_cents_kwh"] = (
    state_retail["total_revenue"] / state_retail["total_mwh"]
) * 0.1

# --- Figure 14: Average price by state ---
fig, ax = plt.subplots(figsize=(12, 8))
for st in STATE_COLORS:
    df_s = state_retail[state_retail["state"] == st].sort_values("report_year")
    if len(df_s) > 0 and df_s["avg_price_cents_kwh"].notna().any():
        ax.plot(
            df_s["report_year"],
            df_s["avg_price_cents_kwh"],
            color=STATE_COLORS[st],
            lw=4,
            label=STATE_NAMES[st],
        )

ax.set_title(
    "Average Retail Electricity Price by State (IOU Aggregate)\nFERC Form 1, 2015-2024",
    fontweight="bold",
)
ax.set_xlabel("Year")
ax.set_ylabel("cents/kWh")
ax.grid(lw=0.3)
ax.legend(fontsize=14)
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "14_avg_price_by_state.png", dpi=150, bbox_inches="tight")
plt.show()


# --- Figure 15: T&D share of total plant ---
td_share = plant_pivot.copy()
if (
    "Transmission" in td_share.columns
    and "Distribution" in td_share.columns
    and "Total Electric Plant" in td_share.columns
):
    td_share["T&D_share"] = (
        (td_share["Transmission"].fillna(0) + td_share["Distribution"].fillna(0))
        / td_share["Total Electric Plant"]
        * 100
    )

    fig, ax = plt.subplots(figsize=(14, 8))
    for uid in UTIL_IDS:
        df_u = td_share[td_share["utility_id_ferc1"] == uid].sort_values("report_year")
        if len(df_u) > 0 and df_u["T&D_share"].notna().any():
            ax.plot(
                df_u["report_year"],
                df_u["T&D_share"],
                color=UTIL_COLORS[uid],
                lw=2.5,
                label=UTIL_NAMES[uid],
            )

    ax.set_title(
        "T&D Share of Total Electric Plant in Service\nFERC Form 1 Schedule 204",
        fontweight="bold",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("% of Total")
    ax.grid(lw=0.3)
    ax.legend(fontsize=9, ncol=3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "15_td_share_of_plant.png", dpi=150, bbox_inches="tight")
    plt.show()


# --- Figure 16: Revenue per MWh by state ---
total_rev_state = (
    total_rev.groupby(["state", "report_year"])["dollar_value"].sum().reset_index()
)
retail_mwh_state = retail_sales.copy()
retail_mwh_state["state"] = retail_mwh_state["utility_id_ferc1"].map(UTIL_STATES)
retail_mwh_state = (
    retail_mwh_state.groupby(["state", "report_year"])["retail_mwh"].sum().reset_index()
)

rev_mwh_state = total_rev_state.merge(
    retail_mwh_state, on=["state", "report_year"]
)
rev_mwh_state["rev_per_mwh"] = (
    rev_mwh_state["dollar_value"] / rev_mwh_state["retail_mwh"]
)

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

ax = axes[0]
for st in STATE_COLORS:
    df_s = rev_mwh_state[rev_mwh_state["state"] == st].sort_values("report_year")
    if len(df_s) > 0:
        ax.plot(
            df_s["report_year"],
            df_s["rev_per_mwh"],
            color=STATE_COLORS[st],
            lw=4,
            label=STATE_NAMES[st],
        )
ax.set_title("Revenue per MWh (Nominal)", fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("$/MWh")
ax.grid(lw=0.3)
ax.legend(fontsize=14)
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

ax = axes[1]
for st in STATE_COLORS:
    df_s = rev_mwh_state[rev_mwh_state["state"] == st].sort_values("report_year")
    if len(df_s) > 0:
        base_val = df_s[df_s["report_year"] == BASE_YEAR]["rev_per_mwh"]
        if not base_val.empty and base_val.item() > 0:
            ax.plot(
                df_s["report_year"],
                df_s["rev_per_mwh"] / base_val.item(),
                color=STATE_COLORS[st],
                lw=4,
                label=STATE_NAMES[st],
            )
ax.axhline(y=1, linestyle="dashed", color="grey", lw=1)
ax.set_title(f"Revenue per MWh Normalized ({BASE_YEAR} = 1.0)", fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel(f"Relative to {BASE_YEAR}")
ax.grid(lw=0.3)
ax.legend(fontsize=14)
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

plt.suptitle(
    "Revenue Requirement per MWh by State\nFERC Form 1, 2015-2024",
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "16_rev_per_mwh_by_state.png", dpi=150, bbox_inches="tight")
plt.show()


# ---------------------------------------------------------------------------
# 9. Summary Tables
# ---------------------------------------------------------------------------
print("\n--- Summary Tables ---")

# Summary for latest year
summary_rows = []
for uid in UTIL_IDS:
    row = {"Utility": UTIL_NAMES[uid], "State": UTIL_STATES[uid]}

    tr = total_rev[
        (total_rev["utility_id_ferc1"] == uid)
        & (total_rev["report_year"] == latest_year)
    ]
    row["Revenue ($B)"] = tr["dollar_value"].sum() / 1e9 if len(tr) > 0 else np.nan

    op = opex_pivot[
        (opex_pivot["utility_id_ferc1"] == uid)
        & (opex_pivot["report_year"] == latest_year)
    ]
    for cat in ["Generation", "Transmission", "Distribution"]:
        row[f"{cat} O&M ($M)"] = (
            op[cat].sum() / 1e6 if (cat in op.columns and len(op) > 0) else np.nan
        )

    pl = plant_pivot[
        (plant_pivot["utility_id_ferc1"] == uid)
        & (plant_pivot["report_year"] == latest_year)
    ]
    row["Total Plant ($B)"] = (
        pl["Total Electric Plant"].sum() / 1e9
        if ("Total Electric Plant" in pl.columns and len(pl) > 0)
        else np.nan
    )

    rs = retail_sales[
        (retail_sales["utility_id_ferc1"] == uid)
        & (retail_sales["report_year"] == latest_year)
    ]
    row["Retail Sales (TWh)"] = rs["retail_mwh"].sum() / 1e6 if len(rs) > 0 else np.nan

    rr = ror_df[
        (ror_df["utility_id_ferc1"] == uid) & (ror_df["report_year"] == latest_year)
    ]
    row["Implied ROR (%)"] = rr["implied_ror"].values[0] if len(rr) > 0 else np.nan

    summary_rows.append(row)

summary = pd.DataFrame(summary_rows).round(2)
print(f"\nSummary for {latest_year}:")
print(summary.to_string(index=False))

# Cumulative % change
pct_rows = []
for uid in UTIL_IDS:
    row = {"Utility": UTIL_NAMES[uid], "State": UTIL_STATES[uid]}

    for cat in ["Generation", "Transmission", "Distribution"]:
        base = opex_pivot[
            (opex_pivot["utility_id_ferc1"] == uid)
            & (opex_pivot["report_year"] == BASE_YEAR)
        ]
        latest = opex_pivot[
            (opex_pivot["utility_id_ferc1"] == uid)
            & (opex_pivot["report_year"] == latest_year)
        ]
        if len(base) > 0 and len(latest) > 0 and cat in base.columns:
            b = base[cat].sum()
            l_val = latest[cat].sum()
            row[f"{cat} O&M %chg"] = (
                ((l_val - b) / abs(b) * 100) if b != 0 else np.nan
            )

    for cat in ["Production", "Transmission", "Distribution"]:
        base = plant_pivot[
            (plant_pivot["utility_id_ferc1"] == uid)
            & (plant_pivot["report_year"] == BASE_YEAR)
        ]
        latest = plant_pivot[
            (plant_pivot["utility_id_ferc1"] == uid)
            & (plant_pivot["report_year"] == latest_year)
        ]
        if len(base) > 0 and len(latest) > 0 and cat in base.columns:
            b = base[cat].sum()
            l_val = latest[cat].sum()
            row[f"{cat} Plant %chg"] = (
                ((l_val - b) / abs(b) * 100) if b != 0 else np.nan
            )

    base_r = total_rev[
        (total_rev["utility_id_ferc1"] == uid)
        & (total_rev["report_year"] == BASE_YEAR)
    ]
    latest_r = total_rev[
        (total_rev["utility_id_ferc1"] == uid)
        & (total_rev["report_year"] == latest_year)
    ]
    if len(base_r) > 0 and len(latest_r) > 0:
        b = base_r["dollar_value"].sum()
        l_val = latest_r["dollar_value"].sum()
        row["Revenue %chg"] = ((l_val - b) / abs(b) * 100) if b != 0 else np.nan

    pct_rows.append(row)

pct_df = pd.DataFrame(pct_rows).round(1)
print(f"\nCumulative % Change: {BASE_YEAR} to {latest_year}")
print(pct_df.to_string(index=False))

# Save to CSV
summary.to_csv(OUTPUT_DIR / "summary_latest_year.csv", index=False)
pct_df.to_csv(OUTPUT_DIR / "cumulative_pct_change.csv", index=False)

print(f"\nAll figures saved to {OUTPUT_DIR}/")
print("Done!")
