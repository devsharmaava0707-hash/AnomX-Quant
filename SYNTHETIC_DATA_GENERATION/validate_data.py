from pathlib import Path
import json

import pandas as pd
import yaml


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file) or {}

DATA_PATH = PROJECT_ROOT / config.get(
    "data", {}
).get("raw_path", "data/raw/events.csv")

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TEXT_REPORT_PATH = REPORT_DIR / "data_validation_report.txt"
JSON_REPORT_PATH = REPORT_DIR / "data_validation_report.json"


# --------------------------------------------------
# 2. Load the dataset
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Run generate_events.py first, or check raw_path in config.yaml."
    )

df = pd.read_csv(DATA_PATH)

results = []
errors = []
warnings = []


def record_check(name, passed, details, severity="error"):
    """Save the result of one validation check."""
    result = {
        "check": name,
        "status": "PASS" if passed else (
            "WARNING" if severity == "warning" else "FAIL"
        ),
        "details": str(details),
    }

    results.append(result)

    if not passed:
        if severity == "warning":
            warnings.append(result)
        else:
            errors.append(result)


# --------------------------------------------------
# 3. Basic data integrity
# --------------------------------------------------

record_check(
    "Dataset is not empty",
    not df.empty,
    f"Rows found: {len(df):,}",
)

record_check(
    "Dataset has columns",
    len(df.columns) > 0,
    f"Columns found: {len(df.columns)}",
)

record_check(
    "Duplicate column names",
    not df.columns.duplicated().any(),
    (
        "No duplicate column names."
        if not df.columns.duplicated().any()
        else f"Duplicates: {df.columns[df.columns.duplicated()].tolist()}"
    ),
)

# Check event IDs if the column exists.
if "event_id" in df.columns:
    duplicate_ids = int(df["event_id"].duplicated().sum())
    missing_ids = int(df["event_id"].isna().sum())

    record_check(
        "Event IDs are unique",
        duplicate_ids == 0,
        f"Duplicate event IDs: {duplicate_ids}",
    )

    record_check(
        "Event IDs are present",
        missing_ids == 0,
        f"Missing event IDs: {missing_ids}",
    )
else:
    record_check(
        "Event ID column exists",
        False,
        "Expected column 'event_id' was not found.",
    )

# Report missing values for every column.
missing_values = df.isna().sum()
columns_with_missing = missing_values[missing_values > 0]

record_check(
    "Required columns are present",
    all(
        column in df.columns
        for column in ["event_id", "timestamp", "is_anomalous"]
    ),
    "Expected: event_id, timestamp, is_anomalous",
)

record_check(
    "Missing values",
    columns_with_missing.empty,
    (
        "No missing values."
        if columns_with_missing.empty
        else columns_with_missing.to_dict()
    ),
    severity="warning",
)


# --------------------------------------------------
# 4. Validate anomaly labels
# --------------------------------------------------

if "is_anomalous" in df.columns:
    labels = pd.to_numeric(df["is_anomalous"], errors="coerce")

    invalid_labels = int(
        (labels.isna() | ~labels.isin([0, 1])).sum()
    )

    record_check(
        "Anomaly labels are valid",
        invalid_labels == 0,
        f"Invalid or missing labels: {invalid_labels}",
    )

    if invalid_labels == 0:
        normal_count = int((labels == 0).sum())
        anomalous_count = int((labels == 1).sum())

        anomaly_percentage = (
            anomalous_count / len(df) * 100 if len(df) else 0
        )

        record_check(
            "Both normal and anomalous events exist",
            normal_count > 0 and anomalous_count > 0,
            (
                f"Normal: {normal_count:,}; "
                f"Anomalous: {anomalous_count:,}; "
                f"Anomalous percentage: {anomaly_percentage:.3f}%"
            ),
            severity="warning",
        )

        if "user_is_anomalous" in df.columns:
            user_labels = pd.to_numeric(
                df["user_is_anomalous"], errors="coerce"
            )
            invalid_user_labels = int(
                (user_labels.isna() | ~user_labels.isin([0, 1])).sum()
            )

            record_check(
                "User anomaly flags are valid",
                invalid_user_labels == 0,
                f"Invalid or missing user flags: {invalid_user_labels}",
            )

else:
    record_check(
        "Anomaly label column exists",
        False,
        "Expected column 'is_anomalous' was not found.",
    )


# --------------------------------------------------
# 5. Validate timestamps and configured date range
# --------------------------------------------------

if "timestamp" in df.columns:
    timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
    invalid_timestamps = int(timestamps.isna().sum())

    record_check(
        "Timestamps can be parsed",
        invalid_timestamps == 0,
        f"Invalid or missing timestamps: {invalid_timestamps}",
    )

    valid_timestamps = timestamps.dropna()

    if not valid_timestamps.empty:
        record_check(
            "Timestamps are chronologically ordered",
            timestamps.is_monotonic_increasing,
            "Checks whether rows are sorted from earliest to latest.",
            severity="warning",
        )

        date_config = config.get("data", {})
        start_date = date_config.get("start_date")
        end_date = date_config.get("end_date")

        if start_date and end_date:
            start = pd.Timestamp(start_date)
            end = pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1)

            outside_range = (
                (valid_timestamps < start)
                | (valid_timestamps >= end)
            )

            outside_count = int(outside_range.sum())

            record_check(
                "Timestamps fall within configured dates",
                outside_count == 0,
                (
                    f"Outside configured range: {outside_count}; "
                    f"configured dates: {start_date} to {end_date}"
                ),
            )

        print("\nTimestamp range:")
        print(f"  Earliest: {valid_timestamps.min()}")
        print(f"  Latest:   {valid_timestamps.max()}")


# --------------------------------------------------
# 6. Check numeric fields
# --------------------------------------------------

for column in [
    "deposit_amount",
    "withdrawal_amount",
    "trade_volume",
]:
    if column in df.columns:
        numeric_values = pd.to_numeric(df[column], errors="coerce")
        invalid_numeric = int(
            (df[column].notna() & numeric_values.isna()).sum()
        )
        negative_values = int((numeric_values < 0).sum())

        record_check(
            f"{column} contains numeric values",
            invalid_numeric == 0,
            f"Non-numeric values: {invalid_numeric}",
        )

        record_check(
            f"{column} is non-negative",
            negative_values == 0,
            f"Negative values: {negative_values}",
        )


# --------------------------------------------------
# 7. Examine anomaly types and behaviour
# --------------------------------------------------

type_column = next(
    (
        column
        for column in ["anomaly_type", "event_type"]
        if column in df.columns
    ),
    None,
)

if type_column:
    print(f"\nEvent/anomaly type distribution ({type_column}):")

    type_counts = (
        df[type_column]
        .fillna("MISSING")
        .value_counts()
        .to_dict()
    )

    for event_type, count in type_counts.items():
        print(f"  {event_type}: {count:,}")

    # Show how many events of each type have anomaly labels.
    if "is_anomalous" in df.columns:
        print("\nAnomaly labels by type:")
        print(
            pd.crosstab(
                df[type_column].fillna("MISSING"),
                df["is_anomalous"],
                dropna=False,
            ).to_string()
        )

        # If normal events use a type such as "normal", do not
        # assume every non-normal type must have label 1.
        # Instead, flag suspicious mismatches for manual review.
        if "anomaly_type" in df.columns:
            normal_names = {"", "normal", "none", "nan"}
            named_as_anomaly = ~(
                df["anomaly_type"]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(normal_names)
            )

            labels = pd.to_numeric(
                df["is_anomalous"], errors="coerce"
            )
            mismatch_count = int(
                (named_as_anomaly & (labels == 0)).sum()
            )

            record_check(
                "Anomaly type and label consistency",
                mismatch_count == 0,
                (
                    f"Rows with a non-normal anomaly_type but label 0: "
                    f"{mismatch_count}. Review these rows; some datasets "
                    "may use anomaly_type for metadata rather than labels."
                ),
                severity="warning",
            )

else:
    record_check(
        "Event/anomaly type column exists",
        False,
        "Neither 'anomaly_type' nor 'event_type' was found.",
        severity="warning",
    )


# --------------------------------------------------
# 8. Compare generated row count with configuration
# --------------------------------------------------

expected_rows = config.get("data", {}).get("n_events")

if expected_rows is not None:
    record_check(
        "Row count matches configuration",
        len(df) == int(expected_rows),
        f"Expected: {int(expected_rows):,}; actual: {len(df):,}",
    )


# --------------------------------------------------
# 9. Print and save the report
# --------------------------------------------------

passed_count = sum(
    result["status"] == "PASS" for result in results
)

report = {
    "dataset_path": str(DATA_PATH),
    "rows": int(len(df)),
    "columns": int(len(df.columns)),
    "checks_passed": int(passed_count),
    "checks_failed": len(errors),
    "warnings": len(warnings),
    "results": results,
}

with open(JSON_REPORT_PATH, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=2, ensure_ascii=False)

with open(TEXT_REPORT_PATH, "w", encoding="utf-8") as file:
    file.write("ANOMX DATA VALIDATION REPORT\n")
    file.write("=" * 40 + "\n")
    file.write(f"Dataset: {DATA_PATH}\n")
    file.write(f"Rows: {len(df):,}\n")
    file.write(f"Columns: {len(df.columns)}\n\n")

    for result in results:
        file.write(
            f"[{result['status']}] {result['check']}\n"
            f"  {result['details']}\n\n"
        )

    file.write(
        f"SUMMARY: {passed_count} passed, "
        f"{len(errors)} failed, {len(warnings)} warnings\n"
    )

print("\n" + "=" * 55)
print("ANOMX DATA VALIDATION SUMMARY")
print("=" * 55)
print(f"Rows:           {len(df):,}")
print(f"Columns:        {len(df.columns)}")
print(f"Checks passed:  {passed_count}")
print(f"Checks failed:  {len(errors)}")
print(f"Warnings:       {len(warnings)}")
print(f"\nText report: {TEXT_REPORT_PATH}")
print(f"JSON report: {JSON_REPORT_PATH}")

if errors:
    print("\nFailed checks:")
    for error in errors:
        print(f"  - {error['check']}: {error['details']}")

print("\nValidation finished.")