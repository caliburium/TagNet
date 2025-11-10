import requests
from pathlib import Path
import sys
import os

FILE_URLS = [
    (
        "https://huggingface.co/datasets/Mike0307/MNIST-M/resolve/main/data/test-00000-of-00001-ba3ad971b105ff65.parquet?download=true",
        "mnistm-test.parquet"
    ),
    (
        "https://huggingface.co/datasets/Mike0307/MNIST-M/resolve/main/data/train-00000-of-00001-571b6b1e2c195186.parquet?download=true",
        "mnistm-train.parquet"
    )
]

TARGET_DIR = Path(__file__).parent / "../data"


def download_file_with_progress(url, destination_path):
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            print(f"'{destination_path.name}' start")

            with open(destination_path, 'wb') as f:
                downloaded_size = 0
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

                    downloaded_size += len(chunk)
                    if total_size > 0:
                        done = int(50 * downloaded_size / total_size)
                        percent = (downloaded_size / total_size) * 100
                        sys.stdout.write(f"\r[{'=' * done}{' ' * (50 - done)}] {percent:.2f}%")
                        sys.stdout.flush()

            sys.stdout.write("\n")
            print(f"download complete: {destination_path.resolve()}")

    except requests.exceptions.RequestException as e:
        print(f"\n'{destination_path.name}' error {e}")
        if destination_path.exists():
            destination_path.unlink()


def main():
    try:
        TARGET_DIR.mkdir(parents=True, exist_ok=True)
        print(f"path: {TARGET_DIR.resolve()}")
    except Exception as e:
        print(f"error ({TARGET_DIR.resolve()}): {e}")
        return

    for url, filename in FILE_URLS:
        destination = TARGET_DIR / filename

        if destination.exists():
            print(f"'{filename}' already exist")
        else:
            download_file_with_progress(url, destination)


if __name__ == "__main__":
    try:
        import requests
    except ImportError:
        sys.exit(1)

    main()