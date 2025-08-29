"""Generate mock CSV data into data/ using the top-level mock_data_generator.py logic.
This wrapper ensures files land under data/ for the app to consume.
"""

from pathlib import Path
import shutil
import runpy

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main():
    # Run existing generator in a temp dir (current working dir)
    # then move generated CSVs to data/
    before = set(p.name for p in ROOT.glob("*.csv"))
    runpy.run_path(str(ROOT / "mock_data_generator.py"))
    after = set(p.name for p in ROOT.glob("*.csv"))
    new_files = [f for f in after - before if f.endswith(".csv")]
    DATA.mkdir(parents=True, exist_ok=True)
    for fname in new_files:
        shutil.move(str(ROOT / fname), str(DATA / fname))
    print(f"Moved {len(new_files)} CSV files into data/.")


if __name__ == "__main__":
    main()
