"""Advanced, causal feature engineering for AnomX synthetic event data.

This script builds behavioral features from timestamped user events while
avoiding obvious target leakage and future-looking rolling statistics.

"""

from __future__ import annotations

import argparse
import logging
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import yaml  # Optional: use the project's config.yaml when available.
except ImportError:  # pragma: no cover - exercised only if PyYAML is absent
    yaml = None


# ---------------------------------------------------------------------------
# 1. Logging and configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("anomx.feature_engineering")

DEFAULT_ROLLING_WINDOWS = [5, 10, 20]
DEFAULT_Z_SCORE_THRESHOLD = 3.0

# These columns describe the answer, the injected anomaly label, or row identity.
# Keep them in the output for auditing, but do not use them as model inputs.
MODEL_EXCLUDE_COLUMNS = {
    "is_anomalous",       # Target y
    "anomaly_type",       # Direct description of injected anomaly
    "anomaly_severity",   # Generator metadata, not an independently observed signal
    "user_is_anomalous",  # User-level label that can leak the event label
    "anomaly_group_id",   # Synthetic grouping/label shortcut
    "event_id",           # Unique identifier
    "user_id",            # Exclude initially to reduce user-identity memorization
    "timestamp",          # Use derived time features instead of the raw datetime
    "event_type",         # Replaced by event_type_* one-hot columns
    "ip_address",         # Raw high-cardinality identifier; use behavior features first
}


def find_project_root() -> Path:
    """Find the project root whether this file sits in the root or a feature folder."""
    here = Path(__file__).resolve().parent
    candidates = [here, here.parent, here.parent.parent]
    for candidate in candidates:
        if (candidate / "configs" / "config.yaml").exists():
            return candidate
        if (candidate / "SYNTHETIC_DATA_GENERATION").exists():
            return candidate
    # Useful when the script is run standalone with --input/--output.
    return here


def load_config(project_root: Path) -> dict[str, Any]:
    """Load config.yaml if possible; defaults keep the script usable standalone."""
    config_path = project_root / "configs" / "config.yaml"
    if yaml is None or not config_path.exists():
        logger.info("No readable configs/config.yaml found; using feature defaults.")
        return {}

    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    return loaded


def resolve_path(value: str | Path, project_root: Path) -> Path:
    """Resolve a configured/CLI path relative to the project root when needed."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def resolve_input_path(
    requested_path: str | None, config: dict[str, Any], project_root: Path
) -> Path:
    """Find raw CSV using CLI, config, or common AnomX locations."""
    if requested_path:
        path = resolve_path(requested_path, project_root)
        if not path.is_file():
            raise FileNotFoundError(f"Input CSV does not exist: {path}")
        return path

    configured_path = config.get("data", {}).get("raw_path")
    candidates: list[Path] = []
    if configured_path:
        candidates.append(resolve_path(configured_path, project_root))

    candidates.extend(
        [
            project_root / "data" / "raw" / "events.csv",
            project_root / "SYNTHETIC_DATA_GENERATION" / "events.csv",
            project_root / "events.csv",
        ]
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    attempted = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Could not locate the raw events CSV. Tried:\n" + attempted
        + "\nPass its location with --input."
    )


def resolve_output_path(
    requested_path: str | None,
    config: dict[str, Any],
    project_root: Path,
) -> Path:
    """Resolve output path from CLI, config, or a sensible default."""
    configured_path = config.get("data", {}).get("processed_path")
    chosen = requested_path or configured_path or "data/processed/events_features.csv"
    return resolve_path(chosen, project_root)


def get_rolling_windows(config: dict[str, Any]) -> list[int]:
    """Read event-count windows from config, falling back to [5, 10, 20]."""
    values = config.get("features", {}).get("rolling_windows", DEFAULT_ROLLING_WINDOWS)
    if not isinstance(values, (list, tuple)):
        logger.warning("features.rolling_windows must be a list; using defaults.")
        return DEFAULT_ROLLING_WINDOWS.copy()

    windows: set[int] = set()
    for value in values:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            logger.warning("Ignoring invalid rolling window: %r", value)
            continue
        if parsed > 0:
            windows.add(parsed)
    return sorted(windows) or DEFAULT_ROLLING_WINDOWS.copy()


def get_z_threshold(config: dict[str, Any]) -> float:
    value = config.get("features", {}).get("z_score_threshold", DEFAULT_Z_SCORE_THRESHOLD)
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        logger.warning("Invalid z_score_threshold; using %.1f", DEFAULT_Z_SCORE_THRESHOLD)
        return DEFAULT_Z_SCORE_THRESHOLD
    return threshold if threshold > 0 else DEFAULT_Z_SCORE_THRESHOLD


# ---------------------------------------------------------------------------
# 2. Data checks and basic calendar features
# ---------------------------------------------------------------------------


def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    """Check essential columns and convert timestamps/numeric fields safely."""
    required = {"user_id", "event_type", "timestamp"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Raw dataset is missing required columns: {missing}")
    if df.empty:
        raise ValueError("The raw dataset has no rows.")

    result = df.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    invalid_timestamps = int(result["timestamp"].isna().sum())
    if invalid_timestamps:
        raise ValueError(
            f"Found {invalid_timestamps} invalid/missing timestamps; fix them before feature engineering."
        )

    # Convert only known numeric measurements. Event-specific missing values remain missing.
    numeric_columns = [
        "hour_of_day", "day_of_week", "is_weekend", "login_success",
        "failed_attempts", "timezone_gap_hours", "lot_size", "trade_volume",
        "pnl", "margin_used", "trade_duration_seconds",
        "trade_volume_vs_baseline", "is_night_trade", "amount",
        "is_immediate_withdrawal", "session_duration_mins", "page_clicks",
        "click_rate_per_min", "account_age_days",
    ]
    for column in numeric_columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    # Stable sorting is important when two events have the same timestamp.
    result["_original_row_order"] = np.arange(len(result), dtype=np.int64)
    result = result.sort_values(
        ["user_id", "timestamp", "_original_row_order"], kind="mergesort"
    )
    return result


def add_calendar_features(df: pd.DataFrame) -> None:
    """Add calendar features, including circular representations of time."""
    hour = df["timestamp"].dt.hour.astype(float)
    day = df["timestamp"].dt.dayofweek.astype(float)

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["day_of_week_sin"] = np.sin(2 * np.pi * day / 7.0)
    df["day_of_week_cos"] = np.cos(2 * np.pi * day / 7.0)
    df["is_night_hour"] = hour.isin([0, 1, 2, 3, 4, 5]).astype("int8")

    # Event type has a small number of categories, so one-hot encoding is practical.
    event_type_dummies = pd.get_dummies(
        df["event_type"].astype("string"), prefix="event_type", dtype="int8"
    )
    for column in event_type_dummies.columns:
        df[column] = event_type_dummies[column]


# ---------------------------------------------------------------------------
# 3. Causal time-gap and time-window activity features
# ---------------------------------------------------------------------------


def add_time_gap_features(df: pd.DataFrame) -> None:
    """Calculate gaps to prior events only; current events are excluded."""
    logger.info("Building time-gap features...")
    previous_event = df.groupby("user_id", sort=False)["timestamp"].shift(1)
    df["time_since_last_event_sec"] = (
        df["timestamp"] - previous_event
    ).dt.total_seconds()

    for event_type in ["login", "trade", "deposit", "withdrawal", "session"]:
        matching_timestamp = df["timestamp"].where(df["event_type"].eq(event_type))
        last_seen = matching_timestamp.groupby(df["user_id"], sort=False).ffill()
        last_seen_before_current = last_seen.groupby(df["user_id"], sort=False).shift(1)
        df[f"time_since_last_{event_type}_sec"] = (
            df["timestamp"] - last_seen_before_current
        ).dt.total_seconds()


def prior_window_counts(
    df: pd.DataFrame,
    source_mask: pd.Series,
    window_seconds: int,
    output_name: str,
) -> None:
    """Count matching events in [current_time-window, current_time), per user.

    The current event and events with exactly the same timestamp are excluded.
    This conservative definition avoids using same-time events as prior knowledge.
    """
    result = pd.Series(np.nan, index=df.index, dtype="float64")
    window_ns = int(window_seconds * 1_000_000_000)

    for _, group in df.groupby("user_id", sort=False):
        times_ns = group["timestamp"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        is_source = source_mask.loc[group.index].fillna(False).to_numpy(dtype=bool)
        source_times = times_ns[is_source]  # Already sorted within each user.

        left = np.searchsorted(source_times, times_ns - window_ns, side="left")
        right = np.searchsorted(source_times, times_ns, side="left")
        result.loc[group.index] = right - left

    df[output_name] = result.astype("int32")


def add_activity_window_features(df: pd.DataFrame) -> None:
    logger.info("Building prior activity-window counts...")
    event_type = df["event_type"].astype("string")
    login_mask = event_type.eq("login")
    trade_mask = event_type.eq("trade")
    deposit_mask = event_type.eq("deposit")
    withdrawal_mask = event_type.eq("withdrawal")

    masks_and_windows = [
        (pd.Series(True, index=df.index), 5 * 60, "prior_event_count_5min"),
        (pd.Series(True, index=df.index), 30 * 60, "prior_event_count_30min"),
        (trade_mask, 5 * 60, "prior_trade_count_5min"),
        (trade_mask, 30 * 60, "prior_trade_count_30min"),
        (login_mask, 10 * 60, "prior_login_count_10min"),
        (deposit_mask, 60 * 60, "prior_deposit_count_1hour"),
        (withdrawal_mask, 60 * 60, "prior_withdrawal_count_1hour"),
    ]
    for mask, seconds, name in masks_and_windows:
        prior_window_counts(df, mask, seconds, name)

    if "login_success" in df.columns:
        login_success = pd.to_numeric(df["login_success"], errors="coerce")
        failed_login_mask = login_mask & login_success.eq(0)
        prior_window_counts(
            df, failed_login_mask, 10 * 60, "failed_logins_last_10min"
        )


# ---------------------------------------------------------------------------
# 4. Rolling features from previous trades and sessions
# ---------------------------------------------------------------------------


def add_prior_rolling_statistics(
    df: pd.DataFrame, windows: list[int]
) -> None:
    """Add per-user rolling statistics using previous same-type events only.

    These features are populated on their relevant event type. For example,
    prior trade statistics are NaN on login rows because trade volume does not
    apply to a login. NaN is deliberately not blanket-filled with zero.
    """
    logger.info("Building prior rolling trade/session statistics...")

    specs = [
        ("trade", "trade_volume", "mean", "trade_volume_mean", 1),
        ("trade", "trade_volume", "std", "trade_volume_std", 2),
        ("trade", "pnl", "mean", "pnl_mean", 1),
        ("trade", "lot_size", "mean", "lot_size_mean", 1),
        ("session", "click_rate_per_min", "mean", "click_rate_mean", 1),
        ("session", "session_duration_mins", "mean", "session_duration_mean", 1),
    ]

    for event_name, column, aggregation, feature_stub, min_periods in specs:
        if column not in df.columns:
            continue
        subset = df.loc[df["event_type"].eq(event_name), ["user_id", column]].copy()
        if subset.empty:
            continue
        values = pd.to_numeric(subset[column], errors="coerce")

        for window in windows:
            shifted = values.groupby(subset["user_id"], sort=False).shift(1)
            rolled = shifted.groupby(subset["user_id"], sort=False).transform(
                lambda series: getattr(
                    series.rolling(window=window, min_periods=min_periods), aggregation
                )()
            )
            feature_name = f"prior_{window}_{event_name}s_{feature_stub}"
            df[feature_name] = np.nan
            df.loc[subset.index, feature_name] = rolled.to_numpy()


# ---------------------------------------------------------------------------
# 5. Login history features: distinct IP/country/device and recent failures
# ---------------------------------------------------------------------------


def add_login_history_features(df: pd.DataFrame) -> None:
    """Summarize the preceding ten logins for every event belonging to a user."""
    logger.info("Building login-history features...")
    required = {"event_type", "login_success"}
    if not required.issubset(df.columns):
        logger.warning("Skipping login history features: missing %s", sorted(required - set(df.columns)))
        return

    categorical_specs = [
        ("ip_address", "unique_ips_last_10_logins"),
        ("country", "unique_countries_last_10_logins"),
        ("device", "unique_devices_last_10_logins"),
    ]
    available_specs = [(source, target) for source, target in categorical_specs if source in df.columns]
    for _, target in available_specs:
        df[target] = np.nan
    df["rolling_failed_logins_last_5_logins"] = np.nan

    for _, group in df.groupby("user_id", sort=False):
        histories = {source: deque(maxlen=10) for source, _ in available_specs}
        failure_history: deque[float] = deque(maxlen=5)

        for row_index, row in group.iterrows():
            # Calculate features before adding the current event to history.
            for source, target in available_specs:
                valid_values = [value for value in histories[source] if pd.notna(value)]
                df.at[row_index, target] = len(set(valid_values)) if histories[source] else np.nan

            valid_failures = [value for value in failure_history if pd.notna(value)]
            df.at[row_index, "rolling_failed_logins_last_5_logins"] = (
                float(sum(valid_failures)) if failure_history else np.nan
            )

            if row["event_type"] == "login":
                for source, _ in available_specs:
                    histories[source].append(row.get(source, np.nan))

                success = pd.to_numeric(pd.Series([row["login_success"]]), errors="coerce").iloc[0]
                if pd.isna(success):
                    failure_history.append(np.nan)
                else:
                    failure_history.append(float(success == 0))


# ---------------------------------------------------------------------------
# 6. Deposit/withdrawal context features
# ---------------------------------------------------------------------------


def add_financial_flow_features(df: pd.DataFrame, history_size: int = 5) -> None:
    """Compare withdrawals with the user's previous five deposits, not future ones."""
    logger.info("Building deposit/withdrawal context features...")
    if "amount" not in df.columns:
        logger.warning("Skipping financial flow features: 'amount' column is absent.")
        return

    df["prior_5_deposit_amount_sum"] = np.nan
    df["withdrawal_to_prior_deposit_ratio"] = np.nan

    for _, group in df.groupby("user_id", sort=False):
        previous_deposits: deque[float] = deque(maxlen=history_size)

        for row_index, row in group.iterrows():
            prior_sum = float(sum(previous_deposits))
            df.at[row_index, "prior_5_deposit_amount_sum"] = prior_sum

            amount = pd.to_numeric(pd.Series([row["amount"]]), errors="coerce").iloc[0]
            if row["event_type"] == "withdrawal" and pd.notna(amount) and prior_sum > 0:
                df.at[row_index, "withdrawal_to_prior_deposit_ratio"] = float(amount / prior_sum)

            # Update history only after building this row's features.
            if row["event_type"] == "deposit" and pd.notna(amount):
                previous_deposits.append(float(amount))


# ---------------------------------------------------------------------------
# 7. Causal per-user z-scores and extreme-value flags
# ---------------------------------------------------------------------------


def add_prior_zscore(
    df: pd.DataFrame,
    event_type: str,
    value_column: str,
    feature_name: str,
    threshold: float,
    min_history: int = 3,
) -> None:
    """Compare the current value with the user's prior same-type observations."""
    if value_column not in df.columns:
        return

    mask = df["event_type"].eq(event_type)
    subset = df.loc[mask, ["user_id", value_column]].copy()
    if subset.empty:
        return

    current = pd.to_numeric(subset[value_column], errors="coerce")
    prior_mean = current.groupby(subset["user_id"], sort=False).transform(
        lambda series: series.shift(1).expanding(min_periods=min_history).mean()
    )
    prior_std = current.groupby(subset["user_id"], sort=False).transform(
        lambda series: series.shift(1).expanding(min_periods=min_history).std(ddof=0)
    )

    # If historical variance is zero, z-score is undefined; preserve NaN.
    z_score = (current - prior_mean) / prior_std.where(prior_std > 0)
    df[feature_name] = np.nan
    df.loc[subset.index, feature_name] = z_score.to_numpy()

    flag_name = f"{feature_name}_extreme"
    flags = pd.Series(np.nan, index=subset.index, dtype="float64")
    known = z_score.notna()
    flags.loc[known] = (z_score.loc[known].abs() >= threshold).astype(float)
    df[flag_name] = np.nan
    df.loc[subset.index, flag_name] = flags.to_numpy()


def add_zscore_features(df: pd.DataFrame, threshold: float) -> None:
    logger.info("Building causal z-score features (history only)...")
    # Amounts apply to deposits and withdrawals, so treat them as one financial-event stream.
    financial_mask = df["event_type"].isin(["deposit", "withdrawal"])
    if "amount" in df.columns and financial_mask.any():
        subset = df.loc[financial_mask, ["user_id", "amount"]].copy()
        current = pd.to_numeric(subset["amount"], errors="coerce")
        prior_mean = current.groupby(subset["user_id"], sort=False).transform(
            lambda series: series.shift(1).expanding(min_periods=3).mean()
        )
        prior_std = current.groupby(subset["user_id"], sort=False).transform(
            lambda series: series.shift(1).expanding(min_periods=3).std(ddof=0)
        )
        z_score = (current - prior_mean) / prior_std.where(prior_std > 0)
        df["amount_zscore"] = np.nan
        df.loc[subset.index, "amount_zscore"] = z_score.to_numpy()
        flag = pd.Series(np.nan, index=subset.index, dtype="float64")
        known = z_score.notna()
        flag.loc[known] = (z_score.loc[known].abs() >= threshold).astype(float)
        df["amount_zscore_extreme"] = np.nan
        df.loc[subset.index, "amount_zscore_extreme"] = flag.to_numpy()

    add_prior_zscore(df, "trade", "trade_volume", "trade_vol_zscore", threshold)
    add_prior_zscore(df, "trade", "pnl", "pnl_zscore", threshold)
    add_prior_zscore(
        df, "session", "session_duration_mins", "session_duration_zscore", threshold
    )


# ---------------------------------------------------------------------------
# 8. Full pipeline
# ---------------------------------------------------------------------------


def build_features(
    raw_df: pd.DataFrame,
    rolling_windows: list[int] | None = None,
    z_threshold: float = DEFAULT_Z_SCORE_THRESHOLD,
) -> pd.DataFrame:
    """Return raw columns plus engineered behavioral features.

    The target and anomaly metadata are retained for audit/evaluation. The
    caller must exclude MODEL_EXCLUDE_COLUMNS when preparing model inputs.
    """
    logger.info("Starting AnomX feature engineering...")
    windows = rolling_windows or DEFAULT_ROLLING_WINDOWS.copy()
    df = validate_input(raw_df)
    original_columns = set(raw_df.columns)

    add_calendar_features(df)
    add_time_gap_features(df)
    add_activity_window_features(df)
    add_prior_rolling_statistics(df, windows)
    add_login_history_features(df)
    add_financial_flow_features(df)
    add_zscore_features(df, z_threshold)

    # Return to source-row order; this preserves the original event file's order.
    df = df.sort_values("_original_row_order", kind="mergesort")
    df = df.drop(columns=["_original_row_order"])

    # Check target is unchanged and no rows were accidentally added/dropped.
    if len(df) != len(raw_df):
        raise RuntimeError(f"Row count changed unexpectedly: {len(raw_df)} -> {len(df)}")
    if "is_anomalous" in raw_df.columns and not df["is_anomalous"].equals(raw_df["is_anomalous"]):
        raise RuntimeError("The target column is not unchanged; refusing to save features.")

    new_columns = [column for column in df.columns if column not in original_columns]
    logger.info(
        "Feature engineering finished: %s rows, %s original columns, %s new features, %s total columns.",
        len(df), len(original_columns), len(new_columns), len(df.columns),
    )
    return df


def run_pipeline(input_path: Path, output_path: Path, config: dict[str, Any]) -> pd.DataFrame:
    logger.info("Loading raw events from %s", input_path)
    raw_df = pd.read_csv(input_path, low_memory=False)
    logger.info("Raw dataset shape: %s", raw_df.shape)

    windows = get_rolling_windows(config)
    z_threshold = get_z_threshold(config)
    logger.info("Rolling windows (previous matching events): %s", windows)
    logger.info("Extreme z-score threshold: %.2f", z_threshold)

    feature_df = build_features(raw_df, windows, z_threshold)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output_path, index=False)

    new_columns = [column for column in feature_df.columns if column not in raw_df.columns]
    model_exclusions_present = [
        column for column in sorted(MODEL_EXCLUDE_COLUMNS) if column in feature_df.columns
    ]
    categorical_columns = [
        column for column in ["country", "device", "instrument", "method", "kyc_change_type"]
        if column in feature_df.columns
    ]

    print("\nFeature engineering completed.")
    print(f"Input file          : {input_path}")
    print(f"Output file         : {output_path}")
    print(f"Rows                : {len(raw_df):,}")
    print(f"Original columns    : {len(raw_df.columns)}")
    print(f"New features        : {len(new_columns)}")
    print(f"Total columns       : {len(feature_df.columns)}")
    print(f"Rolling windows     : {windows} prior matching events")
    print(f"Z-score threshold   : {z_threshold}")

    if "is_anomalous" in feature_df.columns:
        print("\nTarget distribution (retained for training/evaluation, not an input):")
        print(feature_df["is_anomalous"].value_counts(dropna=False).sort_index().to_string())

    print("\nNew features created:")
    for column in new_columns:
        print(f"  + {column}")

    print("\nColumns to exclude from X (model inputs):")
    print("  " + ", ".join(model_exclusions_present))
    if categorical_columns:
        print("\nRaw categorical columns still requiring an explicit modelling decision:")
        print("  " + ", ".join(categorical_columns))

    print("\nNote: NaN values are intentionally retained where a feature is not applicable or history is insufficient.")
    return feature_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AnomX behavioral features from raw event data.")
    parser.add_argument("--input", help="Path to raw events CSV; otherwise config/common paths are tried.")
    parser.add_argument("--output", help="Path for engineered CSV; otherwise config/default path is used.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = find_project_root()
    config = load_config(project_root)
    input_path = resolve_input_path(args.input, config, project_root)
    output_path = resolve_output_path(args.output, config, project_root)

    if input_path == output_path:
        raise ValueError("Input and output paths are identical; refusing to overwrite the raw dataset.")

    run_pipeline(input_path, output_path, config)


if __name__ == "__main__":
    main()
