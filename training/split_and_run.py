import argparse
from pathlib import Path
import csv
import subprocess

DATA_DIR = Path("newfinaldata")
CHUNK_DIR = Path("temp_chunk")
CHUNK_SIZE = 100_000


def split_file(store: str) -> list[Path]:
    path = DATA_DIR / f"processed_{store}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    parts = []
    with path.open("r", encoding="utf-8") as src:
        reader = csv.reader(src)
        header = next(reader)
        chunk_idx = 0
        while True:
            part = CHUNK_DIR / f"{store}_part_{chunk_idx}.csv"
            rows_written = 0
            with part.open("w", encoding="utf-8", newline="") as out:
                writer = csv.writer(out)
                writer.writerow(header)
                for _ in range(CHUNK_SIZE):
                    try:
                        writer.writerow(next(reader))
                        rows_written += 1
                    except StopIteration:
                        break
            if rows_written == 0:
                part.unlink(missing_ok=True)
                break
            parts.append(part)
            chunk_idx += 1
            if rows_written < CHUNK_SIZE:
                break
    return parts


def run_chunks(store: str):
    parts = split_file(store)
    for part in parts:
        print(f"Processing chunk {part}")
        subprocess.run(
            [
                "python",
                "train_lgbm_baseline.py",
                "--stores",
                store,
                "--chunk-file",
                str(part),
            ],
            check=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", required=True)
    args = parser.parse_args()
    run_chunks(args.store)


if __name__ == "__main__":
    main()
