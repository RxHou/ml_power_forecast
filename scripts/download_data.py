import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from power_forecast.config import RAW_DIR, RAW_TXT_PATH, RAW_ZIP_PATH, UCI_ZIP_URLS


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_TXT_PATH.exists():
        print(f"Raw text file already exists: {RAW_TXT_PATH}")
        return

    if not RAW_ZIP_PATH.exists() or RAW_ZIP_PATH.stat().st_size == 0:
        last_error = None
        for url in UCI_ZIP_URLS:
            try:
                print(f"Downloading UCI dataset from {url}")
                download_with_curl(url, RAW_ZIP_PATH)
                print(f"Saved zip file to {RAW_ZIP_PATH}")
                break
            except Exception as exc:
                last_error = exc
                print(f"Download failed: {exc}")
                if RAW_ZIP_PATH.exists():
                    RAW_ZIP_PATH.unlink()
        else:
            raise RuntimeError("All UCI download URLs failed.") from last_error
    else:
        print(f"Zip file already exists: {RAW_ZIP_PATH}")

    print("Extracting dataset...")
    with zipfile.ZipFile(RAW_ZIP_PATH, "r") as zf:
        zf.extractall(RAW_DIR)

    if not RAW_TXT_PATH.exists():
        raise FileNotFoundError(
            f"Expected {RAW_TXT_PATH} after extraction. Check the UCI zip contents."
        )

    print(f"Ready: {RAW_TXT_PATH}")


def download_with_curl(url: str, output_path: Path) -> None:
    command = [
        "curl.exe",
        "-L",
        "--fail",
        "--retry",
        "10",
        "--retry-delay",
        "5",
        "--connect-timeout",
        "30",
        "-C",
        "-",
        "-o",
        str(output_path),
        url,
    ]
    try:
        subprocess.run(command, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        urlretrieve(url, output_path)


if __name__ == "__main__":
    main()
