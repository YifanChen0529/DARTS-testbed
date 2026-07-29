#!/usr/bin/env python3
"""Unified notifier for sweep monitoring. Supports ntfy.sh and Gmail SMTP."""
from __future__ import annotations

import argparse
import os
import smtplib
import sys
from email.message import EmailMessage
from urllib import request as urlreq


def notify_ntfy(topic_url: str, title: str, message: str, priority: str = "default") -> bool:
    try:
        req = urlreq.Request(
            topic_url,
            data=message.encode("utf-8"),
            headers={"Title": title, "Priority": priority, "Tags": "robot"},
        )
        with urlreq.urlopen(req, timeout=10) as resp:
            return 200 <= resp.status < 300
    except Exception as e:
        sys.stderr.write(f"[notify] ntfy fail: {e}\n")
        return False


def notify_smtp(sender: str, password: str, recipient: str, title: str, message: str) -> bool:
    try:
        msg = EmailMessage()
        msg["Subject"] = title
        msg["From"] = sender
        msg["To"] = recipient
        msg.set_content(message)
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as s:
            s.login(sender, password)
            s.send_message(msg)
        return True
    except Exception as e:
        sys.stderr.write(f"[notify] smtp fail: {e}\n")
        return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--title", required=True)
    p.add_argument("--message", required=True)
    p.add_argument("--priority", default="default", choices=["min", "low", "default", "high", "urgent"])
    args = p.parse_args()
    ntfy = os.environ.get("NTFY_TOPIC_URL", "").strip()
    smtp_from = os.environ.get("SMTP_FROM", "").strip()
    smtp_pass = os.environ.get("SMTP_PASS", "").strip()
    smtp_to = os.environ.get("SMTP_TO", "").strip()
    ok = False
    if ntfy:
        ok = notify_ntfy(ntfy, args.title, args.message, args.priority)
    elif smtp_from and smtp_pass and smtp_to:
        ok = notify_smtp(smtp_from, smtp_pass, smtp_to, args.title, args.message)
    else:
        sys.stderr.write("[notify] no backend configured (set NTFY_TOPIC_URL or SMTP_FROM/PASS/TO)\n")
        sys.exit(2)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
