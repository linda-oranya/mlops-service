from __future__ import annotations
import os, json, sys
from pathlib import Path
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/model.pkl"))
LOG_PATH = Path("artifacts/request_log.jsonl") 
REPORT_PATH = Path("artifacts/evidently_report.html")


def build_report(reference_csv: str, current_csv: str, out_html: str = str(REPORT_PATH)):
    ref = pd.read_csv(reference_csv)
    cur = pd.read_csv(current_csv)
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)
    report.save_html(out_html)
    print("Saved drift report to", out_html)


if __name__ == "__main__":
    build_report(sys.argv[1], sys.argv[2])