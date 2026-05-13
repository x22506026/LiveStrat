"""Quick health check for the transaction-cost sensitivity endpoint.

This is a developer-helper, not part of the unittest suite. Run from the
project root with ``python tests\\smoke_endpoints.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import app


def main() -> None:
    client = app.test_client()
    resp = client.get("/api/transaction-cost-sensitivity?timeframe=4h")
    data = resp.get_json()
    print(f"/api/transaction-cost-sensitivity status: {resp.status_code}")
    print(f"Cost sensitivity rows: {len(data.get('rows', []))}")


if __name__ == "__main__":
    main()
