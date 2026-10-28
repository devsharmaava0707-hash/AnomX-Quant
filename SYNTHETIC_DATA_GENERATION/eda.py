
from pathlib import Path

import pandas as pd
import yaml
import matplotlib.pyplot as plt


# ==================================================
# 1. Paths and configuration
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file) or {}

DATA_PATH = PROJECT_ROOT / config.get("data", {}).get(
    "raw_path", "data/raw/events.csv"
)

EDA_DIR = PROJECT_ROOT / "reports" / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. Load data
# ==================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Run generate_events.py first."
    )

df = pd.read_csv(DATA_PATH, low_memory=False)

required_columns = {"timestamp", "is_anomalous", "event_type"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Dataset is missing required columns: {sorted(missing_columns)}"
    )

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["is_anomalous"] = pd.to_numeric(
    df["is_anomalous"], errors="coerce"
)

# Use only rows with valid timestamps and binary event labels
valid_mask = (
    df["timestamp"].notna()
    & df["is_anomalous"].isin([0, 1])
)
analysis_df = df.loc[valid_mask].copy()

if analysis_df.empty:
    raise ValueError("No rows with valid timestamps and anomaly labels.")

analysis_df["is_anomalous"] = (
    analysis_df["is_anomalous"].astype(int)
)

print("=" * 60)
print("ANOMX — EXPLORATORY DATA ANALYSIS")
print("=" * 60)
print(f"Dataset rows:       {len(df):,}")
print(f"Rows used in EDA:   {len(analysis_df):,}")
print(f"Columns:            {len(df.columns)}")
print(f"Output directory:   {EDA_DIR}")


# ==================================================
# 3. Basic dataset summary
# ==================================================

summary_lines = [
    "ANOMX EXPLORATORY DATA ANALYSIS REPORT",
    "=" * 45,
    f"Dataset path: {DATA_PATH}",
    f"Total rows: {len(df):,}",
    f"Rows used in EDA: {len(analysis_df):,}",
    f"Columns: {len(df.columns)}",
    f"Rows excluded due to invalid timestamp or label: "
    f"{len(df) - len(analysis_df):,}",
    "",
    "COLUMN DATA TYPES",
    "-" * 25,
    df.dtypes.to_string(),
    "",
    "MISSING VALUES BY COLUMN",
    "-" * 25,
    df.isna().sum().to_string(),
    "",
    "NUMERICAL SUMMARY",
    "-" * 25,
    df.describe(include="number").to_string(),
    "",
]

normal_count = int((analysis_df["is_anomalous"] == 0).sum())
anomaly_count = int((analysis_df["is_anomalous"] == 1).sum())
anomaly_rate = anomaly_count / len(analysis_df) * 100

summary_lines.extend([
    "ANOMALY SUMMARY",
    "-" * 25,
    f"Normal events: {normal_count:,}",
    f"Anomalous events: {anomaly_count:,}",
    f"Anomaly rate: {anomaly_rate:.3f}%",
    "",
])

print(f"\nNormal events:     {normal_count:,}")
print(f"Anomalous events:  {anomaly_count:,}")
print(f"Anomaly rate:      {anomaly_rate:.3f}%")


# ==================================================
# 4. Plot normal vs anomalous events
# ==================================================

label_counts = (
    analysis_df["is_anomalous"]
    .value_counts()
    .reindex([0, 1], fill_value=0)
)

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(
    ["Normal", "Anomalous"],
    label_counts.values,
)

ax.set_title("Normal vs Anomalous Events")
ax.set_ylabel("Number of events")
ax.bar_label(bars, fmt="%.0f")
fig.tight_layout()
fig.savefig(EDA_DIR / "01_anomaly_distribution.png", dpi=150)
plt.close(fig)


# ==================================================
# 5. Plot event type distribution
# ==================================================

event_counts = analysis_df["event_type"].fillna(
    "Missing"
).value_counts()

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(event_counts.index.astype(str), event_counts.values)
ax.set_title("Event Type Distribution")
ax.set_xlabel("Event type")
ax.set_ylabel("Number of events")
ax.tick_params(axis="x", rotation=35)
ax.bar_label(bars, fmt="%.0f")
fig.tight_layout()
fig.savefig(EDA_DIR / "02_event_type_distribution.png", dpi=150)
plt.close(fig)

summary_lines.extend([
    "EVENT TYPE COUNTS",
    "-" * 25,
    event_counts.to_string(),
    "",
])


# ==================================================
# 6. Plot anomaly type distribution
# ==================================================

if "anomaly_type" in analysis_df.columns:
    anomalous_rows = analysis_df[
        analysis_df["is_anomalous"] == 1
    ]

    anomaly_types = anomalous_rows["anomaly_type"].fillna(
        "Missing"
    ).value_counts()

    if not anomaly_types.empty:
        fig, ax = plt.subplots(figsize=(11, 7))
        bars = ax.barh(
            anomaly_types.index.astype(str)[::-1],
            anomaly_types.values[::-1],
        )

        ax.set_title("Anomalous Events by Anomaly Type")
        ax.set_xlabel("Number of events")
        ax.bar_label(bars, fmt="%.0f", padding=3)
        fig.tight_layout()
        fig.savefig(
            EDA_DIR / "03_anomaly_type_distribution.png",
            dpi=150,
        )
        plt.close(fig)

        summary_lines.extend([
            "ANOMALY TYPE COUNTS",
            "-" * 25,
            anomaly_types.to_string(),
            "",
        ])


# ==================================================
# 7. Compare numerical features by label
# ==================================================

numeric_candidates = [
    "deposit_amount",
    "withdrawal_amount",
    "trade_volume",
    "account_age_days",
]

numeric_columns = [
    column for column in numeric_candidates
    if column in analysis_df.columns
]

for column in numeric_columns:
    analysis_df[column] = pd.to_numeric(
        analysis_df[column], errors="coerce"
    )

    normal_values = analysis_df.loc[
        analysis_df["is_anomalous"] == 0, column
    ].dropna()

    anomaly_values = analysis_df.loc[
        analysis_df["is_anomalous"] == 1, column
    ].dropna()

    # A log scale can help visualize highly skewed amounts.
    # Negative values are excluded from this plot only.
    normal_values = normal_values[normal_values >= 0]
    anomaly_values = anomaly_values[anomaly_values >= 0]

    if normal_values.empty or anomaly_values.empty:
        continue

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.boxplot(
        [normal_values, anomaly_values],
        tick_labels=["Normal", "Anomalous"],
        showfliers=False,
    )

    ax.set_title(f"{column}: Normal vs Anomalous")
    ax.set_ylabel(column)
    fig.tight_layout()

    filename = f"04_{column}_comparison.png"
    fig.savefig(EDA_DIR / filename, dpi=150)
    plt.close(fig)

    stats = analysis_df.groupby("is_anomalous")[column].agg(
        ["count", "mean", "median", "std"]
    )

    summary_lines.extend([
        f"{column.upper()} BY ANOMALY LABEL",
        "-" * 25,
        stats.to_string(),
        "",
    ])


# ==================================================
# 8. Examine event activity over time
# ==================================================

daily_counts = (
    analysis_df.set_index("timestamp")
    .groupby("is_anomalous")
    .resample("D")
    .size()
    .unstack(level=0, fill_value=0)
)

fig, ax = plt.subplots(figsize=(12, 5))

if 0 in daily_counts.columns:
    ax.plot(
        daily_counts.index,
        daily_counts[0],
        label="Normal events",
    )

if 1 in daily_counts.columns:
    ax.plot(
        daily_counts.index,
        daily_counts[1],
        label="Anomalous events",
    )

ax.set_title("Daily Event Counts Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Number of events")
ax.legend()
fig.tight_layout()
fig.savefig(EDA_DIR / "05_daily_activity.png", dpi=150)
plt.close(fig)


# ==================================================
# 9. Hour-of-day activity
# ==================================================

analysis_df["hour"] = analysis_df["timestamp"].dt.hour

hourly_counts = (
    analysis_df.groupby(["hour", "is_anomalous"])
    .size()
    .unstack(fill_value=0)
    .reindex(range(24), fill_value=0)
)

fig, ax = plt.subplots(figsize=(10, 5))

if 0 in hourly_counts.columns:
    ax.plot(
        hourly_counts.index,
        hourly_counts[0],
        marker="o",
        label="Normal",
    )

if 1 in hourly_counts.columns:
    ax.plot(
        hourly_counts.index,
        hourly_counts[1],
        marker="o",
        label="Anomalous",
    )

ax.set_title("Event Activity by Hour of Day")
ax.set_xlabel("Hour (0–23)")
ax.set_ylabel("Number of events")
ax.set_xticks(range(0, 24, 2))
ax.legend()
fig.tight_layout()
fig.savefig(EDA_DIR / "06_hourly_activity.png", dpi=150)
plt.close(fig)


# ==================================================
# 10. Save report
# ==================================================

summary_lines.extend([
    "TIME RANGE",
    "-" * 25,
    f"Earliest event: {analysis_df['timestamp'].min()}",
    f"Latest event: {analysis_df['timestamp'].max()}",
    "",
    "INTERPRETATION NOTES",
    "-" * 25,
    "- EDA describes the generated dataset; it does not prove realism.",
    "- Box plots hide outliers to make the central distributions easier to compare.",
    "- Daily counts show event volume, not necessarily unique users.",
    "- Large differences between classes may be caused by the generator.",
    "- Investigate possible label leakage before model training.",
])

REPORT_PATH = EDA_DIR / "eda_summary.txt"
REPORT_PATH.write_text(
    "\n".join(summary_lines),
    encoding="utf-8",
)

print("\nEDA completed successfully.")
print(f"Charts and report saved in: {EDA_DIR}")
print(f"Summary report: {REPORT_PATH}")
