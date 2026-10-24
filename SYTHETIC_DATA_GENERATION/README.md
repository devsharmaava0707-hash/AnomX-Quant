# Synthetic Data Generation

This module creates a synthetic dataset of user activity for the AnomX anomaly-detection project. It simulates ordinary activity and several deliberately unusual patterns so that later stages can develop and evaluate anomaly-detection methods.

> **Important:** These are hand-designed synthetic patterns for experimentation. They are not real financial records, and an event marked as anomalous is not proof of fraud or misconduct.

## Contents

- [Purpose](#purpose)
- [Files](#files)
- [Configuration](#configuration)
- [How generation works](#how-generation-works)
- [Function reference](#function-reference)
  - [Utility functions and constants](#utility-functions-and-constants)
  - [User profiles](#1-build_user_profiles)
  - [Base and normal events](#2-empty_base-and-generate_normal_event)
  - [Anomaly generators](#3-anomaly-generators)
  - [Generator registry](#4-anomaly_generators)
  - [Dataset orchestration](#5-generate_dataset)
- [Anomaly patterns](#anomaly-patterns)
- [Output](#output)
- [Run the generator](#run-the-generator)
- [Current limitations and future improvements](#current-limitations-and-future-improvements)

## Purpose

The generator creates event records for simulated users. Depending on the event, a record may describe a login, trade, deposit, withdrawal, session, or KYC change. The generator also adds unusual patterns intended to resemble behaviours that an anomaly-detection system might investigate.

The resulting CSV is an input to later AnomX components, such as feature engineering, anomaly detection, and evaluation.

## Files

The current component is organised as follows:

```text
SYNTHETIC_DATA_GENERATION/
├── generate_events.py
├── events.csv
└── README.md
```

- `generate_events.py` contains the configuration loading, helper functions, user-profile creation, normal-event generation, anomaly generators, and dataset orchestration.
- `events.csv` is the generated dataset.
- `README.md` documents this component.

The generator reads `configs/config.yaml` from the project. Keep the path in the code consistent with the actual project structure when moving files.

## Configuration

The current configuration is:

```yaml
data:
  random_seed: 42
  n_users: 1000
  n_events: 50000
  anomaly_fraction: 0.05
  raw_path: data/raw/events.csv
```

| Setting | Meaning |
|---|---|
| `random_seed` | Seed used to make random generation more repeatable, provided all random sources are seeded consistently. |
| `n_users` | Number of synthetic user profiles to create. |
| `n_events` | Target number of event rows in the output. The actual number can exceed this target if anomaly generators create more events than the target allows. |
| `anomaly_fraction` | Fraction used to calculate how many profiles are designated anomalous. With `1000` users and `0.05`, this gives `50` profiles. |
| `raw_path` | Output path for the generated CSV. |

**Reproducibility note:** Python's `random` module and NumPy's random generator are separate sources of randomness. To make results repeatable, seed both consistently. Reproducibility can also depend on library versions and code changes.

## How generation works

At a high level, the generator follows this sequence:

1. Load settings from `config.yaml`.
2. Build user profiles.
3. Select the profiles designated for an anomaly pattern.
4. Generate the configured anomaly events for those profiles.
5. Generate additional ordinary-style events until the target row count is reached, if possible.
6. Convert all event dictionaries into a pandas DataFrame.
7. Sort the records by timestamp and reset the DataFrame index.
8. Create the output directory if necessary and save the DataFrame to CSV.
9. Print a short summary and return the DataFrame.

The main entry point is guarded by:

```python
if __name__ == "__main__":
    generate_dataset()
```

This means `generate_dataset()` runs when the file is executed directly, but not merely when it is imported as a module.

## Function reference

### Utility functions and constants

The file also contains constants and helper functions used by the generators.

#### Configuration loading

The script reads YAML configuration and extracts values such as the random seed, user count, event target, anomaly fraction, and output path. Keeping these values in `config.yaml` makes it easier to change dataset settings without editing the generation logic.

#### `COUNTRY_TIMEZONES`

A mapping between country identifiers and their associated time-zone information. It is used by login-related patterns to simulate users connecting from different countries and to estimate differences in local time zones.

#### `ANOMALY_TYPES`

The list of anomaly-pattern names used when assigning an anomaly type to a profile. The names must match the keys used in `ANOMALY_GENERATORS`.

#### `SHARED_IPS`

A collection of IP addresses used by relevant simulated events. These are synthetic values for the dataset and should not be interpreted as real indicators of malicious activity.

#### `timezone_offset(...)`

A helper used to obtain or calculate a time-zone offset for a country. The exact behaviour depends on the implementation in `generate_events.py`.

#### `random_timestamp(...)`

A helper that generates a random timestamp within the configured simulation period. It lets event generators place activity at different points in the dataset's time range.

#### `random_ip(...)`

A helper that returns a synthetic IP address for an event. Depending on the implementation, it can choose from a pool of addresses or create an address dynamically.

### 1. `build_user_profiles()`

**Purpose:** Create the simulated user population before any events are generated.

**What it does:**

- Iterates over the configured number of users.
- Creates a profile for each user, including a user ID and behavioural attributes used by event generators.
- Assigns a typical trading volume and other profile information.
- Marks a calculated fraction of profiles as anomalous and assigns each such profile an `anomaly_type`.
- Stores profile fields such as a last-active timestamp where defined by the implementation.

For the current configuration, `int(1000 * 0.05)` equals `50`. If the implementation uses `i < n_anomalous`, the first 50 generated profiles are designated anomalous; this is deterministic ordering, not a random selection of 50 users.

**Returns:** A collection of user profiles (the precise container type is defined in the code).

**Why it matters:** Profiles provide a consistent source of user-specific attributes. Normal events can vary by profile, while anomaly generators can deliberately deviate from those attributes.

### 2. `empty_base(user_id, ts)` and `generate_normal_event(...)`

#### `empty_base(user_id, ts)`

**Purpose:** Create a common starting dictionary for an event.

It sets shared fields such as the user ID and timestamp and provides default values—often `None`—for fields that do not apply to every event type.

**Why it matters:** A common event structure makes it easier to combine logins, trades, deposits, withdrawals, sessions, and KYC events into one DataFrame. It also avoids having every generator independently define all common fields.

#### `generate_normal_event(user_id, profile, ts=None)`

**Purpose:** Generate one ordinary-style event for a user.

**What it does:**

1. Chooses an event type using `EVENT_TYPES` and `EVENT_WEIGHTS`.
2. Uses the user's profile and a timestamp (generated if one is not supplied).
3. Starts from the common structure returned by `empty_base(...)`.
4. Fills fields relevant to the chosen event type, such as login details, trade details, deposit/withdrawal information, session activity, or KYC changes.

The weights control how likely each event type is to be selected. They do not necessarily mean that the final dataset will contain those exact percentages, especially after anomaly events are added.

**Returns:** A dictionary representing one event.

**Important distinction:** “Normal” here means generated through the ordinary-event function. It does not guarantee that the event is realistic or statistically normal.

### 3. Anomaly generators

Each anomaly generator creates a sequence of events designed to represent one specific pattern. Most use `generate_normal_event(...)` to establish the common event fields, then update selected fields to create the desired behaviour and set the anomaly labels.

The `n` argument usually controls how many events the function creates, although some functions produce paired events or have a fixed sequence.

#### `events_ip_hopper(user_id, profile, n=8)`

Creates a sequence of login events with changing IP addresses, countries, and devices. It also considers the time-zone difference between the current and previous country.

**Pattern represented:** Logins from changing geographic locations that may be difficult to reconcile with the timing of the activity.

**Caveat:** The generated timestamps are based on separate random increments and are not necessarily guaranteed to be chronological.

#### `events_wash_trader(user_id, profile, n=10)`

Creates a sequence of trades with unusually high volume relative to the user's typical trading volume, alongside other deliberately unusual trading attributes such as lot size, margin, positive PnL, and short durations.

**Pattern represented:** Trading activity that may warrant investigation for wash-trading-like behaviour.

**Caveat:** The generator creates a simplified pattern. It does not establish that trades are matched between related accounts or that actual wash trading has occurred.

#### `events_deposit_withdrawal_cycler(user_id, profile, n=4)`

Creates repeated deposit-and-withdrawal pairs. Each pair contains a deposit followed by a withdrawal after a delay; the withdrawal is generated as a high proportion of the deposited amount.

With the current default `n=4`, the function produces four pairs, or eight events.

**Pattern represented:** Funds moving into and out of an account in a repeated, short-cycle pattern.

**Caveat:** Similar deposit/withdrawal sequences can have legitimate explanations. The pattern is a simulation, not a verdict.

#### `events_bot_trader(user_id, profile, n=8)`

Creates short sessions with unusually high click counts and click rates. The simulated session hour is set to one of a small set of early-morning hours, and the device is set to a consistent browser/platform value.

**Pattern represented:** Highly repetitive or unusually intense session activity that might be associated with automation.

**Caveat:** Setting the hour on a timestamp does not necessarily guarantee that every generated session remains within the intended hours after other timestamp handling. High click rates alone do not prove that a user is a bot.

#### `events_structurer(user_id, profile, n=12)`

Creates repeated deposits with amounts in a relatively narrow range below 1,000, using payment methods such as card or crypto.

**Pattern represented:** Repeated transactions that could be examined for structuring-like behaviour.

**Caveat:** The value of 1,000 is a synthetic threshold chosen by the generator, not a universal legal or regulatory threshold. Amounts below a threshold are not inherently suspicious.

#### `events_brute_forcer(user_id, profile, n=6)`

Creates a series of failed login attempts followed by a successful login. The failed attempts are closely spaced and the failed-attempt count escalates across the sequence.

With the default `n=6`, the intended sequence contains five failed attempts followed by one successful attempt.

**Pattern represented:** Repeated authentication failures followed by a successful login.

**Caveat:** Real systems need additional context—such as source reputation, device history, and authentication logs—to determine whether this pattern is malicious.

#### `events_dormant_withdrawer(user_id, profile, n=3)`

Creates a short sequence of large withdrawals near the end of the configured simulation period.

**Pattern represented:** Significant withdrawals from an account that is intended to represent a dormant-withdrawal scenario.

**Implementation limitation:** The current implementation chooses a `dormant_end` timestamp near the end of the date range and generates withdrawals around it. It does not yet verify a true period of inactivity by comparing the withdrawal timestamp with the user's last actual event. A profile's account age is not the same thing as the duration of dormancy.

#### `events_consistent_winner(user_id, profile, n=10)`

Creates repeated trades with high volumes, positive PnL, large lot sizes/margin, and very short trade durations.

**Pattern represented:** An intentionally unusual sequence of consistently profitable, high-volume trades.

**Caveat:** Consistent profits alone do not prove abuse or latency arbitrage. A robust detector would need additional context and a carefully validated definition of the target behaviour.

#### `events_device_switcher(user_id, profile, n=8)`

Creates login events at regular intervals, varying the selected device while keeping the home IP and country consistent.

**Pattern represented:** Frequent device changes from an otherwise consistent network/geographic context.

**Caveat:** The device is chosen randomly, so the same device may appear more than once. The current generator does not guarantee that every event uses a unique device.

#### `events_kyc_manipulator(user_id, profile, n=4)`

Creates repeated pairs of KYC-change and withdrawal events. A withdrawal follows each KYC change after a short delay.

With the default `n=4`, the function creates four pairs, or eight events.

**Pattern represented:** Account-detail changes followed by withdrawals.

**Caveat:** KYC updates followed by withdrawals can be legitimate. This synthetic sequence is useful for testing a detection pipeline, not for making real-world accusations.

### 4. `ANOMALY_GENERATORS`

**Purpose:** Map each anomaly-type name to the Python function that generates it.

Conceptually, it works like this:

```python
ANOMALY_GENERATORS = {
    "some_anomaly_type": events_some_anomaly,
    # Other anomaly type -> generator function mappings
}
```

The actual keys and function names are defined in `generate_events.py`.

Instead of writing a separate `if`/`elif` branch for every anomaly type, the dataset generator can retrieve the appropriate function using the profile's `anomaly_type` and call it.

**Important:** Every `anomaly_type` assigned to a profile must have a corresponding key in this dictionary. The strings must match exactly.

### 5. `generate_dataset()`

**Purpose:** Coordinate the full generation process and write the final dataset.

**Main responsibilities:**

1. Build user profiles.
2. Identify profiles designated for anomaly generation.
3. Select the matching anomaly function through `ANOMALY_GENERATORS`.
4. Add generated anomaly events to the event collection.
5. Generate additional ordinary-style events until the target event count is reached, where possible.
6. Build a pandas DataFrame from the event dictionaries.
7. Sort events by timestamp and reset the row index.
8. Create the output directory if necessary.
9. Save the DataFrame as a CSV file.
10. Print a summary and return the DataFrame.

**Returns:** The generated pandas DataFrame.

**Important count detail:** If the anomaly generators create more events than `n_events`, the number of output rows may exceed the configured target. The current approach does not automatically trim excess anomaly events.

**Important labelling detail:** The current implementation may label additional ordinary-style events from profiles designated as anomalous as anomalous. This means an anomalous *user* and an anomalous *event* can be conflated. A future improvement should define event-level labels separately and assign them according to the event's actual generation logic.

## Anomaly patterns

| Generator | Main synthetic behaviour | Default output count |
|---|---|---:|
| `events_ip_hopper` | Rapidly changing login locations/IPs/devices | 8 events |
| `events_wash_trader` | Unusually large, short-duration trades | 10 events |
| `events_deposit_withdrawal_cycler` | Deposit followed by a high-proportion withdrawal | 8 events |
| `events_bot_trader` | Short, high-activity sessions | 8 events |
| `events_structurer` | Repeated deposits below a synthetic threshold | 12 events |
| `events_brute_forcer` | Failed logins followed by success | 6 events |
| `events_dormant_withdrawer` | Large withdrawals near the simulation end | 3 events |
| `events_consistent_winner` | Repeated high-volume, profitable short trades | 10 events |
| `events_device_switcher` | Repeated logins with varying devices | 8 events |
| `events_kyc_manipulator` | KYC changes followed by withdrawals | 8 events |

Counts shown are the functions' default outputs; total dataset counts also depend on how many profiles are assigned to each anomaly type and how the dataset orchestration calls the functions.

## Output

The generator writes a CSV to the path specified by `data.raw_path` in `config.yaml`. In the current configuration, that path is `data/raw/events.csv`, relative to the working directory used when running the script.

The CSV is intended to contain a common set of columns for different event types. Fields that do not apply to a particular event may be empty/`None`.

To inspect the output in Python:

```python
import pandas as pd

df = pd.read_csv("data/raw/events.csv")

print(df.head())
print(df.shape)
print(df.columns.tolist())
print(df["event_type"].value_counts(dropna=False))
```

The column names shown by `df.columns` are the authoritative list for the current implementation.

## Run the generator

Run commands from the project root, where `configs/config.yaml` is available.

### 1. Activate the virtual environment (Windows PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

If the project has a `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
```

The generator uses Python libraries including pandas, NumPy, and PyYAML. Python's standard library provides modules such as `random`, `datetime`, and `pathlib`.

### 3. Execute the script

The command depends on the script's actual location. If it is currently inside `SYNTHETIC_DATA_GENERATION/`, run:

```powershell
python .\SYNTHETIC_DATA_GENERATION\generate_events.py
```

If you later move it, update the command and verify that configuration and output paths still resolve correctly.

## Current limitations and future improvements

The synthetic generator is an initial test-data source, not a validated financial-crime simulator. Useful improvements include:

1. **Separate user-level and event-level labels.** Track whether a profile is designated anomalous separately from whether each individual event is anomalous.
2. **Make anomaly-user selection random.** Select the required number of profiles using a seeded random sample rather than always designating the first profiles.
3. **Enforce the event-count target.** Decide explicitly whether `n_events` is a strict total or a target that anomaly generation may exceed.
4. **Model genuine dormancy.** Track the user's last real activity and generate a withdrawal only after a defined inactive period.
5. **Improve timestamp consistency.** Ensure sequences are ordered where required and define whether timestamps are UTC or local time.
6. **Validate generated records.** Check required fields, valid event types, positive amounts where appropriate, and consistent labels.
7. **Add automated tests.** Test event counts, schema, reproducibility, and the intended behaviour of each generator.
8. **Document assumptions and calibrate patterns.** Thresholds and behaviours should be configurable and clearly identified as synthetic assumptions.

These improvements can be implemented incrementally as the AnomX pipeline develops.
