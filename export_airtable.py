#!/usr/bin/env python3
"""Export Airtable table data to CSV and archive locally.

Required environment variables:
  AIRTABLE_PAT
  AIRTABLE_BASE_ID
  AIRTABLE_TABLE_NAME

Optional environment variables:
  AIRTABLE_VIEW                (default: "")
  AIRTABLE_PAGE_SIZE           (default: 100, max 100)
  AIRTABLE_API_URL             (default: https://api.airtable.com/v0)
  OUTPUT_CSV_PATH              (default: webapp submisssions.csv)
  ARCHIVE_DIR                  (default: archive)
  ARCHIVE_PREFIX               (default: webapp_submissions)
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required env var: {name}")
    return value


def _optional_env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def fetch_airtable_records(
    pat: str,
    base_id: str,
    table_name: str,
    view: str,
    page_size: int,
    api_url: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    offset = ""

    while True:
        params: dict[str, Any] = {"pageSize": page_size}
        if view:
            params["view"] = view
        if offset:
            params["offset"] = offset

        endpoint = (
            f"{api_url.rstrip('/')}/{quote(base_id)}/{quote(table_name)}"
            f"?{urlencode(params)}"
        )
        req = Request(
            endpoint,
            headers={
                "Authorization": f"Bearer {pat}",
                "Content-Type": "application/json",
            },
            method="GET",
        )

        with urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        batch = payload.get("records", [])
        if not isinstance(batch, list):
            raise ValueError("Unexpected Airtable response: 'records' is not a list")
        records.extend(batch)

        offset = payload.get("offset", "")
        if not offset:
            break

    return records


def records_to_rows(records: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    field_names: set[str] = set()
    for rec in records:
        fields = rec.get("fields", {})
        if isinstance(fields, dict):
            field_names.update(fields.keys())

    ordered_fields = sorted(field_names)
    headers = ["record_id", "created_time", *ordered_fields]
    rows: list[dict[str, Any]] = []

    for rec in records:
        fields = rec.get("fields", {}) if isinstance(rec.get("fields"), dict) else {}
        row: dict[str, Any] = {
            "record_id": rec.get("id", ""),
            "created_time": rec.get("createdTime", ""),
        }
        for key in ordered_fields:
            value = fields.get(key, "")
            # Keep nested structures readable in CSV.
            if isinstance(value, (dict, list)):
                row[key] = json.dumps(value, ensure_ascii=False)
            else:
                row[key] = value
        rows.append(row)

    return headers, rows


def write_csv(path: Path, headers: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def archive_csv(source_csv: Path, archive_dir: Path, archive_prefix: str) -> Path:
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    archive_path = archive_dir / f"{archive_prefix}_{stamp}.csv"
    shutil.copy2(source_csv, archive_path)
    return archive_path


def main() -> int:
    try:
        pat = _required_env("AIRTABLE_PAT")
        base_id = _required_env("AIRTABLE_BASE_ID")
        table_name = _required_env("AIRTABLE_TABLE_NAME")

        view = _optional_env("AIRTABLE_VIEW", "")
        page_size = int(_optional_env("AIRTABLE_PAGE_SIZE", "100"))
        api_url = _optional_env("AIRTABLE_API_URL", "https://api.airtable.com/v0")
        output_csv = Path(_optional_env("OUTPUT_CSV_PATH", "webapp submisssions.csv"))
        archive_dir = Path(_optional_env("ARCHIVE_DIR", "archive"))
        archive_prefix = _optional_env("ARCHIVE_PREFIX", "webapp_submissions")

        page_size = max(1, min(page_size, 100))

        print("Fetching Airtable records...")
        records = fetch_airtable_records(
            pat=pat,
            base_id=base_id,
            table_name=table_name,
            view=view,
            page_size=page_size,
            api_url=api_url,
        )
        print(f"Fetched {len(records)} records.")

        headers, rows = records_to_rows(records)
        write_csv(output_csv, headers, rows)
        archive_path = archive_csv(output_csv, archive_dir, archive_prefix)

        print(f"Wrote CSV: {output_csv}")
        print(f"Archived CSV: {archive_path}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

