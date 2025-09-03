#!/usr/bin/env python
from __future__ import annotations
import sys
from adapters.csv_adapter import read_csv_tables, get_last_load_stats

def main() -> int:
    try:
        read_csv_tables()
        stats = get_last_load_stats()
        print("✔ Data validation passed.")
        for name, meta in stats.get("files", {}).items():
            rr = meta.get("rows_read", 0)
            rv = meta.get("rows_valid", 0)
            print(f"  - {name}: {rv}/{rr} valid")
        return 0
    except Exception as e:
        stats = get_last_load_stats()
        print("✖ Data validation failed.")
        if stats.get("errors"):
            print("\nDetails:")
            for msg in stats["errors"]:
                print(f"  - {msg}")
        else:
            # Also surface any attached per-table error
            for name, meta in stats.get("files", {}).items():
                if meta.get("error"):
                    print(f"  - {name}: {meta['error']}")
            print(f"  - {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())