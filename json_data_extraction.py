"""Fetch JSON data from a URL or a local file and optionally persist the result."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

DEFAULT_SOURCE = "https://raw.githubusercontent.com/emredeveloper/Database/main/db.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load JSON data from a remote URL or a local file."
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=DEFAULT_SOURCE,
        help="URL or file path of the JSON resource to load.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional path to save the retrieved JSON data.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Timeout in seconds used for HTTP requests (default: 10).",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON output using indentation.",
    )
    return parser.parse_args()


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"}


def fetch_json_from_url(url: str, *, timeout: float) -> Any:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        raise SystemExit(f"Veri alınırken bir hata oluştu: {exc}") from exc


def load_json_from_file(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise SystemExit(f"Dosya bulunamadı: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"JSON formatı geçersiz: {path} ({exc})") from exc


def save_json(data: Any, destination: Path, *, pretty: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        if pretty:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        else:
            json.dump(data, handle, ensure_ascii=False)


def main() -> None:
    args = parse_args()

    source = args.source
    if is_url(source):
        data = fetch_json_from_url(source, timeout=args.timeout)
    else:
        data = load_json_from_file(Path(source))

    print("Alınan Veri:")
    if args.pretty:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(data, ensure_ascii=False))

    if args.output:
        save_json(data, args.output, pretty=args.pretty)
        print(f"Veri '{args.output}' dosyasına kaydedildi.")


if __name__ == "__main__":
    main()
