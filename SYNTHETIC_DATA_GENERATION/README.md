# AnomX — Synthetic Data Generation

This component generates synthetic user activity for the AnomX anomaly-detection project. It combines ordinary activity with deliberately injected behavioural patterns for testing downstream feature engineering and detection models.

> **Important:** The data is synthetic. A generated anomaly label is ground truth for this simulation only; it is not evidence of real fraud.

## Structure

```text
AnomX_SOC_2026-main/
├── configs/
│   └── config.yaml
├── SYNTHETIC_DATA_GENERATION/
│   ├── generate_events.py
│   └── README.md
└── data/
    └── raw/
        └── events.csv          # generated when the script runs
```

The source file remains a single generator file, as requested. The generator is stored in `SYNTHETIC_DATA_GENERATION` to organize the project by functionality rather than week.

## Configuration

Edit `configs/config.yaml`:

| Key | Purpose |
|---|---|
| `data.random_seed` | Seeds Python `random` and NumPy for repeatable output under the same code and dependency versions. |
| `data.n_users` | Number of simulated user profiles. |
| `data.n_events` | Target number of output events. The generator trims excess injected events if needed and fills remaining rows with ordinary events. |
| `data.anomaly_fraction` | Fraction of profiles designated as anomalous. |
| `data.start_date` / `data.end_date` | Simulation period, as ISO dates (`YYYY-MM-DD`). |
| `data.raw_path` | CSV output path. |

The `anomalies.enabled` list controls which anomaly types can be assigned to anomalous profiles. The generator validates the names and stops with a clear error if a configured type is unknown or the list is empty. Selection among enabled types is random; this does not guarantee that every enabled type appears in every dataset.

## Running the generator

Run from the project root so `configs/config.yaml` resolves correctly.

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
python .\SYNTHETIC_DATA_GENERATION\generate_events.py
```

Install dependencies into your virtual environment if needed:

```powershell
python -m pip install pandas numpy pyyaml
```

The generated CSV will be saved to `data/raw/events.csv`.

## Generation workflow

1. Load configuration and seed random number generators.
2. Create user profiles with home country/IP, preferred device, instruments, typical trade/deposit amounts, and activity attributes.
3. Assign an anomaly pattern to a subset of profiles.
4. Generate the event sequences for those profiles.
5. Generate ordinary events until the requested total is reached.
6. Sort by timestamp, write the CSV, and print event/anomaly summaries.

## Event schema and labels

Common fields include `event_id`, `user_id`, `event_type`, `timestamp`, `hour_of_day`, `day_of_week`, `is_weekend`, and event-specific fields for logins, trades, deposits/withdrawals, sessions, and KYC changes.

Additional fields introduced in this upgrade:

- `user_is_anomalous`: user-level label; indicates that the profile was selected for anomaly injection.
- `is_anomalous`: event-level label; set to 1 for events directly injected by an anomaly generator, and 0 for ordinary generated events.
- `anomaly_group_id`: shared synthetic group identifier for coordinated-account activity where applicable.
- `anomaly_severity`: a qualitative severity assigned to selected new anomaly events.

Keeping `user_is_anomalous` separate from `is_anomalous` prevents all ordinary events from an anomalous profile from automatically being treated as anomalous.

## Existing anomaly patterns

| Pattern | Simulated behaviour |
|---|---|
| `ip_hopper` | Rapid login IP/country/device changes |
| `wash_trader` | Repeated unusually large trades with selected trading attributes |
| `deposit_withdrawal_cycler` | Repeated deposits followed by high-proportion withdrawals |
| `bot_trader` | Very short sessions with high click rates |
| `structurer` | Repeated deposits in a narrow, synthetic amount range |
| `brute_forcer` | Repeated failed logins followed by success |
| `dormant_withdrawer` | Large withdrawals near the simulation end |
| `consistent_winner` | Repeated high-volume profitable trades with short durations |
| `device_switcher` | Repeated logins with changing devices |
| `kyc_manipulator` | KYC changes followed by withdrawals |

## New advanced anomaly patterns

| Pattern | Simulated behaviour |
|---|---|
| `account_takeover` | Unfamiliar device and country, authentication failures, then a large crypto withdrawal |
| `impossible_travel` | Successful logins from multiple countries within minutes |
| `burst_activity` | A cluster of high-volume trades in a very short interval |
| `coordinated_accounts` | Multiple profiles share a synthetic IP, group ID, and a common time anchor |
| `behavioural_drift` | Trade volumes increase gradually above the user's baseline |

These are intentionally simplified signals for experimentation. For example, geographic changes are represented using country/time-zone proxies rather than real geodesic distances, and coordinated activity is simulated rather than inferred from real network data.

## Reliability improvements in this version

- Start and end dates are configurable.
- Normal deposit generation prevents negative amounts.
- Output directory creation follows the configured output path.
- The target row count is enforced by trimming excess injected events or filling the remaining count.
- Event IDs are assigned sequentially after sorting to guarantee uniqueness.
- Enabled anomaly types are configurable and validated.
- Event-level labels are separated from user-level labels.
- A `coordinated_accounts` helper assigns shared group metadata to sets of at least two profiles when enough such profiles are selected.

## Limitations and next improvements

This is an incremental improvement, not a production-grade financial simulator. Important next steps:

1. Make `anomalies.enabled` actually control which patterns can be selected.
2. Make coordinated-account group sizes configurable and guarantee enough coordinated users when that pattern is enabled.
3. Generate unique, deterministic event IDs rather than random IDs that could collide.
4. Make timestamps and timezone semantics explicit (prefer UTC for storage).
5. Ensure every event generator's timestamps stay within the configured period.
6. Add automated schema, label, count, and reproducibility tests.
7. Add summary visualizations and validate that normal and anomalous distributions are distinguishable without being unrealistically obvious.
8. Consider separate labels for anomaly family, severity, and event/user scope.

## Ethics and interpretation

This dataset contains artificial behaviours and thresholds. Patterns such as large withdrawals, frequent device changes, or high trading volume can also be legitimate. The model trained on this dataset must be evaluated carefully and should not be treated as a real-world fraud detector without external validation.
