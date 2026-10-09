import hashlib
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
INPUT_FILE = PROJECT_ROOT / "examples" / "pride.short.txt"
OUTPUT_DIR = PROJECT_ROOT / "output" / "test"
CHECKSUM_FILE = PROJECT_ROOT / "tests" / "fixtures" / "pride.short.sha256"


def _expected_checksums() -> dict[str, str]:
    checksums = {}
    for line in CHECKSUM_FILE.read_text(encoding="utf-8").splitlines():
        checksum, filename = line.split(maxsplit=1)
        checksums[filename] = checksum
    return checksums


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_cli_generates_expected_outputs():
    expected_checksums = _expected_checksums()
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    result = subprocess.run(
        [
            "booknlp",
            str(INPUT_FILE),
            "--output-dir",
            str(OUTPUT_DIR),
            "--model",
            "small"
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    actual_files = {path.name for path in OUTPUT_DIR.iterdir() if path.is_file()}
    assert actual_files == set(expected_checksums)

    actual_checksums = {
        filename: _sha256(OUTPUT_DIR / filename) for filename in actual_files
    }
    assert actual_checksums == expected_checksums
