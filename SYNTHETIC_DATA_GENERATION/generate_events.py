# src/data_gen/generate_events.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import yaml
from pathlib import Path

# ── Load config ───────────────────────────────────────────────────────────────
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

SEED         = config["data"]["random_seed"]
N_USERS      = config["data"]["n_users"]
N_EVENTS     = config["data"]["n_events"]
ANOMALY_FRAC = config["data"]["anomaly_fraction"]
START_DATE = datetime.fromisoformat(config["data"].get("start_date", "2024-01-01"))
END_DATE = datetime.fromisoformat(config["data"].get("end_date", "2024-12-31"))
if END_DATE <= START_DATE:
    raise ValueError("data.end_date must be later than data.start_date")

np.random.seed(SEED)
random.seed(SEED)

# ── Constants ─────────────────────────────────────────────────────────────────
INSTRUMENTS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD", "USDCHF", "AUDUSD"]
COUNTRIES   = ["IN", "US", "UK", "SG", "AE", "NG", "RU", "CN", "DE", "BR"]
DEVICES     = ["chrome_win", "safari_mac", "android_app", "ios_app", "firefox_linux"]
EVENT_TYPES = ["login", "trade", "deposit", "withdrawal", "session", "kyc_change"]
EVENT_WEIGHTS = [0.22, 0.35, 0.15, 0.10, 0.12, 0.06]

# Timezone offsets per country (UTC offset in hours) — used for impossibility detection
COUNTRY_TIMEZONES = {
    "IN": 5.5, "US": -5, "UK": 0, "SG": 8,
    "AE": 4,   "NG": 1,  "RU": 3, "CN": 8,
    "DE": 1,   "BR": -3
}

ANOMALY_TYPES = [
    "ip_hopper",
    "wash_trader",
    "deposit_withdrawal_cycler",
    "bot_trader",
    "structurer",
    "brute_forcer",
    "dormant_withdrawer",
    "consistent_winner",
    "device_switcher",
    "kyc_manipulator",
    "account_takeover",
    "impossible_travel",
    "burst_activity",
    "coordinated_accounts",
    "behavioural_drift",
]

# Hub IPs shared across multiple anomalous users — for network graph anomaly detection
SHARED_IPS = [f"172.16.{i}.1" for i in range(10)]
ENABLED_ANOMALY_TYPES = config.get("anomalies", {}).get("enabled", ANOMALY_TYPES)
unknown_anomalies = sorted(set(ENABLED_ANOMALY_TYPES) - set(ANOMALY_TYPES))
if unknown_anomalies:
    raise ValueError(f"Unknown anomaly types in config: {unknown_anomalies}")
if not ENABLED_ANOMALY_TYPES:
    raise ValueError("At least one anomaly type must be enabled")

def random_timestamp(start=START_DATE, end=END_DATE):
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))

def random_ip():
    return f"192.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"

def timezone_offset(country):
    return COUNTRY_TIMEZONES.get(country, 0)

# ── Build user profiles ───────────────────────────────────────────────────────
def build_user_profiles():
    profiles = {}
    n_anomalous = int(N_USERS * ANOMALY_FRAC)

    for i in range(N_USERS):
        user_id      = f"USER_{i:04d}"
        is_anomalous = i < n_anomalous
        anomaly_type = random.choice(ENABLED_ANOMALY_TYPES) if is_anomalous else None

        if is_anomalous and random.random() < 0.3:
            home_ip = random.choice(SHARED_IPS)
        else:
            home_ip = f"10.{random.randint(0,255)}.{random.randint(0,255)}.1"

        home_country = random.choice(COUNTRIES)

        profiles[user_id] = {
            "home_ip"              : home_ip,
            "home_country"        : home_country,
            "home_timezone_offset": timezone_offset(home_country),
            "preferred_device"    : random.choice(DEVICES),
            "instruments"         : random.sample(INSTRUMENTS, k=random.randint(1, 3)),
            "typical_trade_vol"   : round(np.random.uniform(1000, 30000), 2),
            "typical_deposit"     : round(np.random.uniform(100, 3000), 2),
            "activity_level"      : random.choice(["low", "medium", "high"]),
            "active_hour_start"   : random.randint(6, 10),
            "active_hour_end"     : random.randint(18, 23),
            "is_anomalous"        : is_anomalous,
            "anomaly_type"        : anomaly_type,
            "anomaly_group_id"    : None,
            "account_created_at"  : random_timestamp(
                START_DATE - timedelta(days=365),
                START_DATE
            ),
            "last_active"         : random_timestamp(START_DATE, START_DATE + timedelta(days=30)),
        }

    return profiles

# ── Empty event base ──────────────────────────────────────────────────────────
def empty_base(user_id, ts):
    return {
        "event_id"                    : f"EVT_{random.randint(100000,999999)}",
        "user_id"                     : user_id,
        "event_type"                  : None,
        "timestamp"                   : ts,
        "hour_of_day"                 : ts.hour,
        "day_of_week"                 : ts.weekday(),
        "is_weekend"                  : int(ts.weekday() >= 5),
        "is_anomalous"                : 0,
        "anomaly_type"                : "none",
        # Login fields
        "ip_address"                  : None,
        "country"                     : None,
        "device"                      : None,
        "login_success"               : None,
        "failed_attempts"             : None,
        # ── EXTRA: timezone gap between home country and login country
        "timezone_gap_hours"          : None,
        # Trade fields
        "instrument"                  : None,
        "lot_size"                    : None,
        "trade_volume"                : None,
        "pnl"                         : None,
        "margin_used"                 : None,
        "trade_duration_seconds"      : None,
        # ── EXTRA: ratio of this trade volume vs user's typical
        "trade_volume_vs_baseline"    : None,
        # ── EXTRA: is this a night trade (12am-5am local time)?
        "is_night_trade"              : None,
        # Deposit/withdrawal fields
        "amount"                      : None,
        "method"                      : None,
        # ── EXTRA: withdrawal to deposit ratio signal
        "is_immediate_withdrawal"     : None,
        # Session fields
        "session_duration_mins"       : None,
        "page_clicks"                 : None,
        # ── EXTRA: clicks per minute — bot detection
        "click_rate_per_min"          : None,
        # KYC fields
        "kyc_change_type"             : None,
        # ── EXTRA: account age in days at time of event
        "account_age_days"            : None,
        # Dataset-quality and advanced anomaly metadata
        "user_is_anomalous"           : int(0),
        "anomaly_group_id"             : None,
        "anomaly_severity"             : None,
    }

# ── Normal event generator ────────────────────────────────────────────────────
def generate_normal_event(user_id, profile, ts=None):
    if ts is None:
        ts = random_timestamp()

    event_type = random.choices(EVENT_TYPES, weights=EVENT_WEIGHTS)[0]
    base = empty_base(user_id, ts)
    base["event_type"]       = event_type
    base["account_age_days"] = max(0, (ts - profile["account_created_at"]).days)
    base["user_is_anomalous"] = int(profile.get("is_anomalous", False))

    if event_type == "login":
        country = profile["home_country"]
        tz_gap  = abs(timezone_offset(country) - profile["home_timezone_offset"])
        base.update({
            "ip_address"        : profile["home_ip"],
            "country"           : country,
            "device"            : profile["preferred_device"],
            "login_success"     : 1,
            "failed_attempts"   : int(np.random.choice([0,1], p=[0.95, 0.05])),
            "timezone_gap_hours": tz_gap,
        })

    elif event_type == "trade":
        vol = profile["typical_trade_vol"]
        actual_vol = round(np.random.normal(vol, vol * 0.2), 2)
        local_hour = (ts.hour + profile["home_timezone_offset"]) % 24
        base.update({
            "instrument"                : random.choice(profile["instruments"]),
            "lot_size"                  : round(np.random.uniform(0.01, 2.0), 2),
            "trade_volume"              : actual_vol,
            "pnl"                       : round(np.random.normal(0, 200), 2),
            "margin_used"               : round(np.random.uniform(100, 5000), 2),
            "trade_duration_seconds"    : int(np.random.uniform(60, 3600)),
            "trade_volume_vs_baseline"  : round(actual_vol / vol, 3),
            "is_night_trade"            : int(0 <= local_hour <= 5),
        })

    elif event_type == "deposit":
        base.update({
            "amount"              : max(1.0, round(np.random.normal(profile["typical_deposit"], 200), 2)),
            "method"              : random.choice(["card", "bank", "crypto"]),
            "is_immediate_withdrawal": 0,
        })

    elif event_type == "withdrawal":
        base.update({
            "amount"              : round(np.random.uniform(50, profile["typical_deposit"] * 0.8), 2),
            "method"              : random.choice(["bank", "crypto"]),
            "is_immediate_withdrawal": 0,
        })

    elif event_type == "session":
        duration = round(np.random.uniform(5, 60), 1)
        clicks   = int(np.random.uniform(5, 80))
        base.update({
            "session_duration_mins": duration,
            "page_clicks"          : clicks,
            "device"               : profile["preferred_device"],
            "click_rate_per_min"   : round(clicks / max(duration, 0.1), 2),
        })

    elif event_type == "kyc_change":
        base.update({
            "kyc_change_type": random.choice(["address_update", "phone_update"]),
        })

    return base

# ── Anomalous event generators ────────────────────────────────────────────────

def events_ip_hopper(user_id, profile, n=8):
    """8.1 Rapid IP + country switching — timezone_gap_hours will be high"""
    events = []
    base_ts = random_timestamp()
    prev_country = profile["home_country"]
    for i in range(n):
        ts      = base_ts + timedelta(minutes=random.randint(1, 15))
        country = random.choice(COUNTRIES)
        tz_gap  = abs(timezone_offset(country) - timezone_offset(prev_country))
        e = generate_normal_event(user_id, profile, ts)
        e.update({
            "event_type"        : "login",
            "ip_address"        : random_ip(),
            "country"           : country,
            "device"            : random.choice(DEVICES),
            "login_success"     : 1,
            "failed_attempts"   : 0,
            "timezone_gap_hours": tz_gap,   # ← key signal: large gap = impossible travel
            "is_anomalous"      : 1,
            "anomaly_type"      : "ip_hopper",
        })
        prev_country = country
        events.append(e)
    return events

def events_wash_trader(user_id, profile, n=10):
    """8.3 Volume spike 10-20x baseline, single instrument, always profits"""
    events = []
    base_ts = random_timestamp()
    vol = profile["typical_trade_vol"]
    for i in range(n):
        ts         = base_ts + timedelta(minutes=i * 2)
        actual_vol = round(vol * random.uniform(10, 20), 2)
        e = generate_normal_event(user_id, profile, ts)
        e.update({
            "event_type"               : "trade",
            "instrument"               : INSTRUMENTS[0],
            "lot_size"                 : round(np.random.uniform(20, 50), 2),
            "trade_volume"             : actual_vol,
            "pnl"                      : round(np.random.uniform(500, 5000), 2),
            "margin_used"              : round(np.random.uniform(50000, 200000), 2),
            "trade_duration_seconds"   : int(np.random.uniform(1, 30)),
            "trade_volume_vs_baseline" : round(actual_vol / vol, 3),  # ← will be 10-20x
            "is_night_trade"           : 0,
            "is_anomalous"             : 1,
            "anomaly_type"             : "wash_trader",
        })
        events.append(e)
    return events

def events_deposit_withdrawal_cycler(user_id, profile, n=4):
    """8.2 Deposit → immediate withdrawal, is_immediate_withdrawal=1"""
    events = []
    base_ts = random_timestamp()
    for i in range(n):
        ts_dep = base_ts + timedelta(hours=i * 12)
        dep_amount = round(np.random.uniform(5000, 20000), 2)

        e_dep = generate_normal_event(user_id, profile, ts_dep)
        e_dep.update({
            "event_type"             : "deposit",
            "amount"                 : dep_amount,
            "method"                 : "crypto",
            "is_immediate_withdrawal": 1,
            "is_anomalous"           : 1,
            "anomaly_type"           : "deposit_withdrawal_cycler",
        })
        events.append(e_dep)

        ts_wit = ts_dep + timedelta(hours=random.randint(1, 3))
        e_wit = generate_normal_event(user_id, profile, ts_wit)
        e_wit.update({
            "event_type"             : "withdrawal",
            "amount"                 : round(dep_amount * random.uniform(0.85, 0.98), 2),
            "method"                 : "crypto",
            "is_immediate_withdrawal": 1,
            "is_anomalous"           : 1,
            "anomaly_type"           : "deposit_withdrawal_cycler",
        })
        events.append(e_wit)
    return events

def events_bot_trader(user_id, profile, n=8):
    """8.4 Inhuman click rate, very short sessions at 2-4 AM"""
    events = []
    base_ts = random_timestamp()
    base_ts = base_ts.replace(hour=random.randint(2, 4))
    for i in range(n):
        ts       = base_ts + timedelta(minutes=i * 3)
        duration = round(np.random.uniform(0.2, 1.5), 1)
        clicks   = int(np.random.uniform(300, 600))
        e = generate_normal_event(user_id, profile, ts)
        e.update({
            "event_type"           : "session",
            "session_duration_mins": duration,
            "page_clicks"          : clicks,
            "click_rate_per_min"   : round(clicks / max(duration, 0.1), 2),  # ← 200-3000/min
            "device"               : "chrome_win",
            "hour_of_day"          : ts.hour,
            "is_anomalous"         : 1,
            "anomaly_type"         : "bot_trader",
        })
        events.append(e)
    return events

def events_structurer(user_id, profile, n=12):
    """8.2 Many deposits just under 1000 — classic structuring pattern"""
    events = []
    base_ts = random_timestamp()
    for i in range(n):
        ts = base_ts + timedelta(hours=i * 2)
        e = generate_normal_event(user_id, profile, ts)
        e.update({
            "event_type"  : "deposit",
            "amount"      : round(np.random.uniform(490, 999), 2),
            "method"      : random.choice(["card", "crypto"]),
            "is_anomalous": 1,
            "anomaly_type": "structurer",
        })
        events.append(e)
    return events

def events_brute_forcer(user_id, profile, n=6):
    """8.7 Escalating failed_attempts then sudden login success"""
    events = []
    base_ts = random_timestamp()
    for i in range(n - 1):
        ts = base_ts + timedelta(seconds=i * 30)
        e = generate_normal_event(user_id, profile, ts)
        e.update({
            "event_type"       : "login",
            "ip_address"       : random_ip(),
            "country"          : random.choice(COUNTRIES),
            "login_success"    : 0,
            "failed_attempts"  : i + 1,
            "is_anomalous"     : 1,
            "anomaly_type"     : "brute_forcer",
        })
        events.append(e)

    ts_ok = base_ts + timedelta(seconds=n * 30)
    e_ok  = generate_normal_event(user_id, profile, ts_ok)
    e_ok.update({
        "event_type"      : "login",
        "ip_address"      : random_ip(),
        "login_success"   : 1,
        "failed_attempts" : n - 1,
        "is_anomalous"    : 1,
        "anomaly_type"    : "brute_forcer",
    })
    events.append(e_ok)
    return events

def events_dormant_withdrawer(user_id, profile, n=3):
    """8.2 Long dormancy then sudden large withdrawal — account_age_days will be high"""
    events = []
    dormant_end = END_DATE - timedelta(days=random.randint(1, 5))
    for i in range(n):
        ts = dormant_end + timedelta(hours=i)
        e  = generate_normal_event(user_id, profile, ts)
        e.update({
            "event_type"  : "withdrawal",
            "amount"      : round(np.random.uniform(15000, 50000), 2),
            "method"      : "crypto",
            "is_anomalous": 1,
            "anomaly_type": "dormant_withdrawer",
        })
        events.append(e)
    return events

def events_consistent_winner(user_id, profile, n=10):
    """8.3 Always positive PnL, sub-10s trades — latency arbitrage signal"""
    events = []
    base_ts = random_timestamp()
    vol = profile["typical_trade_vol"]
    for i in range(n):
        ts         = base_ts + timedelta(minutes=i * 5)
        actual_vol = round(np.random.uniform(100000, 500000), 2)
        local_hour = (ts.hour + profile["home_timezone_offset"]) % 24
        e = generate_normal_event(user_id, profile, ts)
        e.update({
            "event_type"               : "trade",
            "instrument"               : random.choice(INSTRUMENTS[:2]),
            "lot_size"                 : round(np.random.uniform(5, 15), 2),
            "trade_volume"             : actual_vol,
            "pnl"                      : round(np.random.uniform(200, 2000), 2),
            "trade_duration_seconds"   : int(np.random.uniform(1, 10)),
            "margin_used"              : round(np.random.uniform(10000, 50000), 2),
            "trade_volume_vs_baseline" : round(actual_vol / vol, 3),
            "is_night_trade"           : int(0 <= local_hour <= 5),
            "is_anomalous"             : 1,
            "anomaly_type"             : "consistent_winner",
        })
        events.append(e)
    return events

def events_device_switcher(user_id, profile, n=8):
    """8.4 Different device fingerprint every login"""
    events = []
    base_ts = random_timestamp()
    for i in range(n):
        ts = base_ts + timedelta(hours=i)
        e  = generate_normal_event(user_id, profile, ts)
        e.update({
            "event_type"      : "login",
            "device"          : random.choice(DEVICES),
            "ip_address"      : profile["home_ip"],
            "country"         : profile["home_country"],
            "login_success"   : 1,
            "failed_attempts" : 0,
            "is_anomalous"    : 1,
            "anomaly_type"    : "device_switcher",
        })
        events.append(e)
    return events

def events_kyc_manipulator(user_id, profile, n=4):
    """8.7 KYC change immediately followed by large withdrawal"""
    events = []
    base_ts = random_timestamp()
    for i in range(n):
        ts = base_ts + timedelta(hours=i * 2)

        e_kyc = generate_normal_event(user_id, profile, ts)
        e_kyc.update({
            "event_type"     : "kyc_change",
            "kyc_change_type": random.choice(["bank_account_update", "address_update", "id_resubmit"]),
            "is_anomalous"   : 1,
            "anomaly_type"   : "kyc_manipulator",
        })
        events.append(e_kyc)

        ts_wit = ts + timedelta(minutes=random.randint(10, 60))  # ← small gap = suspicious
        e_wit  = generate_normal_event(user_id, profile, ts_wit)
        e_wit.update({
            "event_type"  : "withdrawal",
            "amount"      : round(np.random.uniform(10000, 40000), 2),
            "method"      : "crypto",
            "is_anomalous": 1,
            "anomaly_type": "kyc_manipulator",
        })
        events.append(e_wit)
    return events



# ── Advanced anomaly generators ───────────────────────────────────────────────

def events_account_takeover(user_id, profile, n=6):
    """Unfamiliar device/IP followed by failed logins and a sensitive withdrawal."""
    events = []
    base_ts = random_timestamp(START_DATE, END_DATE - timedelta(hours=3))
    foreign_country = random.choice([c for c in COUNTRIES if c != profile["home_country"]])
    suspicious_ip = random_ip()
    new_device = random.choice([d for d in DEVICES if d != profile["preferred_device"]])
    for i in range(max(1, n - 1)):
        ts = base_ts + timedelta(minutes=i * random.randint(1, 4))
        e = generate_normal_event(user_id, profile, ts)
        e.update({
            "event_type": "login", "ip_address": suspicious_ip,
            "country": foreign_country, "device": new_device,
            "login_success": 0 if i < max(1, n - 2) else 1,
            "failed_attempts": i + 1,
            "timezone_gap_hours": abs(timezone_offset(foreign_country) - profile["home_timezone_offset"]),
            "is_anomalous": 1, "anomaly_type": "account_takeover",
            "anomaly_severity": "high",
        })
        events.append(e)
    ts = base_ts + timedelta(minutes=max(1, n) * 4)
    e = generate_normal_event(user_id, profile, ts)
    e.update({
        "event_type": "withdrawal",
        "amount": round(max(profile["typical_deposit"] * 8, np.random.uniform(8000, 30000)), 2),
        "method": "crypto", "ip_address": suspicious_ip, "country": foreign_country,
        "device": new_device, "is_anomalous": 1, "anomaly_type": "account_takeover",
        "anomaly_severity": "critical",
    })
    events.append(e)
    return events


def events_impossible_travel(user_id, profile, n=4):
    """Login from distant countries too close together in time."""
    events = []
    base_ts = random_timestamp(START_DATE, END_DATE - timedelta(hours=1))
    countries = [profile["home_country"]]
    choices = [c for c in COUNTRIES if c != profile["home_country"]]
    random.shuffle(choices)
    countries.extend(choices[:max(1, n - 1)])
    for i, country in enumerate(countries[:n]):
        ts = base_ts + timedelta(minutes=i * random.randint(2, 8))
        e = generate_normal_event(user_id, profile, ts)
        e.update({
            "event_type": "login", "ip_address": random_ip(),
            "country": country, "device": random.choice(DEVICES),
            "login_success": 1, "failed_attempts": 0,
            "timezone_gap_hours": abs(timezone_offset(country) - profile["home_timezone_offset"]),
            "is_anomalous": 1, "anomaly_type": "impossible_travel",
            "anomaly_severity": "high",
        })
        events.append(e)
    return events


def events_burst_activity(user_id, profile, n=12):
    """Many trades compressed into a short time window."""
    events = []
    base_ts = random_timestamp(START_DATE, END_DATE - timedelta(hours=1))
    vol = profile["typical_trade_vol"]
    for i in range(n):
        ts = base_ts + timedelta(seconds=i * random.randint(5, 20))
        actual_vol = round(max(1, np.random.normal(vol * random.uniform(2.0, 5.0), vol * 0.15)), 2)
        e = generate_normal_event(user_id, profile, ts)
        e.update({
            "event_type": "trade", "instrument": random.choice(profile["instruments"]),
            "trade_volume": actual_vol, "trade_volume_vs_baseline": round(actual_vol / max(vol, 1), 3),
            "lot_size": round(random.uniform(1, 8), 2),
            "pnl": round(np.random.normal(0, 800), 2),
            "trade_duration_seconds": random.randint(5, 120),
            "margin_used": round(random.uniform(1000, 20000), 2),
            "is_anomalous": 1, "anomaly_type": "burst_activity",
            "anomaly_severity": "medium",
        })
        events.append(e)
    return events


def events_coordinated_accounts(user_id, profile, n=5):
    """Events from accounts assigned to a shared synthetic group and common IP."""
    events = []
    group_id = profile.get("anomaly_group_id") or f"COORD_{user_id}"
    shared_ip = profile.get("coordinated_ip") or random.choice(SHARED_IPS)
    base_ts = profile.get("coordinated_start") or random_timestamp(START_DATE, END_DATE - timedelta(hours=2))
    for i in range(n):
        ts = base_ts + timedelta(minutes=i * 10 + random.randint(0, 2))
        e = generate_normal_event(user_id, profile, ts)
        event_type = ["deposit", "trade", "withdrawal"][i % 3]
        e.update({
            "event_type": event_type,
            "ip_address": shared_ip,
            "anomaly_group_id": group_id,
            "is_anomalous": 1, "anomaly_type": "coordinated_accounts",
            "anomaly_severity": "high",
        })
        if event_type in ("deposit", "withdrawal"):
            e.update({
                "amount": round(random.uniform(1500, 9000), 2),
                "method": random.choice(["crypto", "bank"]),
            })
        else:
            trade_volume = round(random.uniform(5000, 50000), 2)
            e.update({
                "trade_volume": trade_volume,
                "instrument": random.choice(profile["instruments"]),
                "trade_volume_vs_baseline": round(trade_volume / max(profile["typical_trade_vol"], 1), 3),
            })
        events.append(e)
    return events


def events_behavioural_drift(user_id, profile, n=10):
    """Gradual increase in trade volume rather than one sudden spike."""
    events = []
    base_ts = random_timestamp(START_DATE, END_DATE - timedelta(hours=max(1, n)))
    baseline = profile["typical_trade_vol"]
    for i in range(n):
        multiplier = 1.0 + (3.0 * (i / max(n - 1, 1)))
        actual_vol = round(max(1, np.random.normal(baseline * multiplier, baseline * 0.08)), 2)
        ts = base_ts + timedelta(hours=i)
        e = generate_normal_event(user_id, profile, ts)
        e.update({
            "event_type": "trade", "instrument": random.choice(profile["instruments"]),
            "trade_volume": actual_vol,
            "trade_volume_vs_baseline": round(actual_vol / max(baseline, 1), 3),
            "lot_size": round(random.uniform(0.1, 4), 2),
            "pnl": round(np.random.normal(0, 400), 2),
            "trade_duration_seconds": random.randint(30, 1800),
            "margin_used": round(random.uniform(500, 10000), 2),
            "is_anomalous": 1, "anomaly_type": "behavioural_drift",
            "anomaly_severity": "medium" if i < n // 2 else "high",
        })
        events.append(e)
    return events


def assign_coordinated_groups(profiles, group_size=3):
    """Assign coordinated anomaly profiles to groups sharing an IP and time anchor."""
    ids = [uid for uid, profile in profiles.items() if profile.get("anomaly_type") == "coordinated_accounts"]
    random.shuffle(ids)
    for offset in range(0, len(ids), group_size):
        group = ids[offset:offset + group_size]
        if len(group) < 2:
            # A coordinated pattern needs at least two accounts; reclassify a lone profile.
            for uid in group:
                profiles[uid]["anomaly_type"] = random.choice([
                    "account_takeover", "impossible_travel", "burst_activity", "behavioural_drift"
                ])
            continue
        group_id = f"COORD_{offset // group_size:04d}"
        shared_ip = random.choice(SHARED_IPS)
        start = random_timestamp(START_DATE, END_DATE - timedelta(hours=2))
        for uid in group:
            profiles[uid]["anomaly_group_id"] = group_id
            profiles[uid]["coordinated_ip"] = shared_ip
            profiles[uid]["coordinated_start"] = start
    return profiles


# ── Dispatcher ────────────────────────────────────────────────────────────────
ANOMALY_GENERATORS = {
    "ip_hopper"                 : events_ip_hopper,
    "wash_trader"               : events_wash_trader,
    "deposit_withdrawal_cycler" : events_deposit_withdrawal_cycler,
    "bot_trader"                : events_bot_trader,
    "structurer"                : events_structurer,
    "brute_forcer"              : events_brute_forcer,
    "dormant_withdrawer"        : events_dormant_withdrawer,
    "consistent_winner"         : events_consistent_winner,
    "device_switcher"           : events_device_switcher,
    "kyc_manipulator"           : events_kyc_manipulator,
    "account_takeover"           : events_account_takeover,
    "impossible_travel"          : events_impossible_travel,
    "burst_activity"             : events_burst_activity,
    "coordinated_accounts"       : events_coordinated_accounts,
    "behavioural_drift"          : events_behavioural_drift,
}

# ── Main ──────────────────────────────────────────────────────────────────────
def generate_dataset():
    """Generate a reproducible synthetic dataset with event-level ground-truth labels."""
    print("Building user profiles...")
    profiles = build_user_profiles()
    assign_coordinated_groups(profiles)
    user_ids = list(profiles.keys())
    anomalous_users = [u for u, profile in profiles.items() if profile["is_anomalous"]]

    all_events = []
    print(f"Injecting anomaly patterns for {len(anomalous_users)} users...")
    for uid in anomalous_users:
        profile = profiles[uid]
        generator = ANOMALY_GENERATORS[profile["anomaly_type"]]
        generated = generator(uid, profile)
        # Every injected event has event-level anomaly truth; preserve user-level label separately.
        for event in generated:
            event["user_is_anomalous"] = int(profile["is_anomalous"])
            if event.get("anomaly_severity") is None:
                event["anomaly_severity"] = "high"
            if event.get("anomaly_group_id") is None:
                event["anomaly_group_id"] = profile.get("anomaly_group_id")
        all_events.extend(generated)

    # Fill to target. Do not relabel all events from an anomalous user as anomalous.
    print(f"Generating ordinary events to reach {N_EVENTS} total...")
    if len(all_events) > N_EVENTS:
        print(f"Warning: anomaly injection produced {len(all_events)} events, above target {N_EVENTS}; trimming injected events to target.")
        all_events = all_events[:N_EVENTS]
    remaining = N_EVENTS - len(all_events)
    for _ in range(remaining):
        uid = random.choice(user_ids)
        profile = profiles[uid]
        all_events.append(generate_normal_event(uid, profile))

    df = pd.DataFrame(all_events)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    # Sequential IDs guarantee uniqueness, including across repeated random draws.
    df["event_id"] = [f"EVT_{i:07d}" for i in range(1, len(df) + 1)]
    df["is_anomalous"] = df["is_anomalous"].fillna(0).astype(int)
    df["user_is_anomalous"] = df["user_is_anomalous"].fillna(0).astype(int)

    out_path = Path(config["data"]["raw_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\\n{'='*56}")
    print(f"Dataset saved       : {out_path}")
    print(f"Total events        : {len(df)} (target: {N_EVENTS})")
    print(f"Columns             : {len(df.columns)}")
    print(f"Anomalous users     : {len(anomalous_users)} / {N_USERS}")
    print(f"Anomalous events    : {int(df['is_anomalous'].sum())}")
    print("\\nEvent breakdown:")
    print(df["event_type"].value_counts(dropna=False))
    print("\\nAnomaly breakdown:")
    print(df.loc[df["is_anomalous"] == 1, "anomaly_type"].value_counts(dropna=False))
    print(f"{'='*56}")
    return df

if __name__ == "__main__":
    generate_dataset()