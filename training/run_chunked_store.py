import argparse
import subprocess
from pathlib import Path

DATA_DIR = Path("newfinaldata")
STORE_PREFIX = "processed_"
SUBMISSION = Path("future_finaldata/submission_with_val.csv")


def split_store(store: str, chunk_size: int) -> list[Path]:
    path = DATA_DIR / f"{STORE_PREFIX}{store}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    parts = []
    with path.open("r", encoding="utf-8") as f:
        header = f.readline()
        idx = 0
        while True:
            lines = [header]
            for _ in range(chunk_size):
                line = f.readline()
                if not line:
                    break
                lines.append(line)
            if len(lines) == 1:
                break
            part = Path(f"temp/{store}_part_{idx}.csv")
            part.parent.mkdir(parents=True, exist_ok=True)
            part.write_text("".join(lines), encoding="utf-8")
            parts.append(part)
            idx += 1
            if len(lines) < chunk_size + 1:
                break
    return parts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", required=True)
    parser.add_argument("--chunk_size", type=int, default=1_000_000)
    args = parser.parse_args()
    parts = split_store(args.store, args.chunk_size)
    for part in parts:
        print(f"Processing {part}")
        subprocess.run(
            [
                "python",
                "train_lgbm_baseline.py",
                "--stores",
                args.store,
                "--chunk-file",
                str(part),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
