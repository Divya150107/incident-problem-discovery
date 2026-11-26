"""Synthetic incident tickets WITH free-text descriptions and planted problems.

The UCI log has no descriptions, so text analysis can't be evaluated on it. This
generator gives a labelled set: routine tickets (normal demand) plus planted
problems whose member tickets are known, so grouping methods can be scored on
precision/recall. Descriptions include names/emails/phone numbers so the
PII-scrubbing step is exercised too. Everything here is fictional.

Planted problems are designed to stress category-based grouping:
  P1 spans categories (users log the same fault under Network, Laptop, Software)
  P2 hides inside a busy category full of routine tickets
  P3 is localised to one site, P4 is bursty, P5 is small
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FIRST = ["Asha", "Ravi", "Meena", "John", "Priya", "Karthik", "Sara", "Vikram", "Divya", "Arun", "Lakshmi", "Tom"]
SITES = ["Chennai", "Cheltenham", "Shanghai", "Sao Paulo", "Pune", "Houston"]

ROUTINE = {
    "Access": ["please reset my password for {app}", "account locked out cannot log in to {app}",
               "need access to {app} shared folder", "request new user account for {app}",
               "forgot password {app} login"],
    "Email": ["how do I set up an out of office reply", "please increase my mailbox size",
              "add me to the {team} distribution list", "email signature not showing correctly",
              "cannot find an old email, please help restore"],
    "Hardware": ["laptop keyboard key broken", "need a second monitor", "mouse not working replace please",
                 "laptop battery drains quickly", "request docking station"],
    "Software": ["please install {app} on my laptop", "{app} licence expired", "update {app} to latest version",
                 "{app} running slow today", "uninstall old version of {app}"],
    "Network": ["wifi slow in meeting room", "cannot connect to guest wifi", "network cable port not working at desk",
                "vpn password reset needed", "how to connect to vpn from home"],
    "Printing": ["add printer to my laptop", "printer out of toner", "how to scan to email", "print job stuck in queue"],
}
APPS = ["SAP", "Teams", "Outlook", "Excel", "Power BI", "Salesforce", "ServiceNow"]
TEAMS = ["finance", "engineering", "sales", "hr"]

PLANTED = {
    "P1": {"name": "VPN drops after laptop update",
           "categories": ["Network", "Hardware", "Software"],
           "templates": ["vpn keeps disconnecting every few minutes since the update",
                         "after windows update my vpn connection drops constantly",
                         "globalprotect vpn disconnects randomly after patch last night",
                         "laptop lost vpn after update, reconnects then drops again",
                         "vpn tunnel dropping since latest windows patch, cannot work remotely"],
           "n": 70, "sites": None, "burst": False},
    "P2": {"name": "Outlook crashes opening shared calendar",
           "categories": ["Email"],
           "templates": ["outlook crashes whenever I open the shared team calendar",
                         "outlook freezes and closes when opening a shared calendar",
                         "shared calendar makes outlook crash, not responding",
                         "opening {name}'s shared calendar crashes outlook every time",
                         "outlook not responding then crashes on shared calendars"],
           "n": 60, "sites": None, "burst": False},
    "P3": {"name": "Floor printers offline at one site",
           "categories": ["Printing"],
           "templates": ["printer on 3rd floor shows offline again",
                         "all printers on our floor offline cannot print",
                         "3rd floor printer offline, others in office same issue",
                         "printers offline again this morning on level 3"],
           "n": 35, "sites": ["Chennai"], "burst": False},
    "P4": {"name": "SAP report timeout at period end",
           "categories": ["Software"],
           "templates": ["SAP financial report times out before finishing",
                         "sap month end report timeout error, cannot close books",
                         "SAP report running forever then timeout message",
                         "timeout when generating SAP period end report"],
           "n": 40, "sites": None, "burst": True},
    "P5": {"name": "Badge reader rejects valid cards",
           "categories": ["Hardware"],
           "templates": ["badge reader at main door rejects my card",
                         "door badge not working at entrance, card is valid",
                         "access card rejected by reader at building entrance"],
           "n": 15, "sites": ["Cheltenham"], "burst": False},
}


def _pii(rng) -> str:
    n = rng.choice(FIRST)
    return rng.choice([
        f" - {n} ({n.lower()}.{rng.choice(['k','s','m'])}@example.com)",
        f" call me on +44 7700 9{rng.integers(10000, 99999)}",
        f" regards {n}",
        f" from {n}, ext {rng.integers(1000, 9999)}",
        "", "",
    ])


def _noise(text: str, rng) -> str:
    if rng.random() < 0.15:   # typo
        i = rng.integers(1, max(len(text) - 1, 2))
        text = text[:i] + text[i + 1:]
    if rng.random() < 0.3:
        text = text.capitalize()
    if rng.random() < 0.2:
        text += rng.choice([" urgent", " pls help", " asap", " thanks", "!!"])
    return text


def generate(n_routine: int = 2800, seed: int = 7, start: str = "2026-01-05", days: int = 84) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start)
    rows = []
    cats = list(ROUTINE)
    weights = np.array([0.25, 0.2, 0.15, 0.15, 0.15, 0.10])
    for _ in range(n_routine):
        cat = rng.choice(cats, p=weights)
        txt = rng.choice(ROUTINE[cat]).format(app=rng.choice(APPS), team=rng.choice(TEAMS), name=rng.choice(FIRST))
        rows.append({"category": cat, "site": rng.choice(SITES), "true_problem": "none",
                     "opened_at": t0 + pd.Timedelta(hours=float(rng.uniform(0, days * 24))),
                     "description": _noise(txt, rng) + _pii(rng)})
    for pid, p in PLANTED.items():
        burst_day = t0 + pd.Timedelta(days=int(rng.integers(days // 2, days - 3)))
        for _ in range(p["n"]):
            txt = rng.choice(p["templates"]).format(name=rng.choice(FIRST))
            when = (burst_day + pd.Timedelta(hours=float(rng.uniform(0, 48)))) if p["burst"] \
                else t0 + pd.Timedelta(hours=float(rng.uniform(0, days * 24)))
            rows.append({"category": rng.choice(p["categories"]),
                         "site": rng.choice(p["sites"]) if p["sites"] else rng.choice(SITES),
                         "true_problem": pid, "opened_at": when, "description": _noise(txt, rng) + _pii(rng)})
    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "number", [f"SYN{i:06d}" for i in range(len(df))])
    return df
