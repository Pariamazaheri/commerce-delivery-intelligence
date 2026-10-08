"""Create a clean GitHub upload ZIP and verify its file fingerprints."""

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
files = subprocess.check_output(
    ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
    cwd=ROOT,
    text=True,
).splitlines()
interactive = "reports/interactive/route_explorer.html"
if (ROOT / interactive).exists():
    files.append(interactive)
files = sorted(set(files))
if any(".venv/" in name or name.startswith("data/raw/") for name in files):
    raise ValueError("Unexpected environment/raw data in upload bundle")
manifest = {
    name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files
}
archive_path = ROOT.parent / f"{ROOT.name}.zip"
with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
    for name in files:
        archive.write(ROOT / name, f"{ROOT.name}/{name}")
    archive.writestr(
        f"{ROOT.name}/DELIVERY_MANIFEST.json", json.dumps(manifest, indent=2)
    )
with zipfile.ZipFile(archive_path) as archive:
    assert archive.testzip() is None
    for name, expected in manifest.items():
        assert (
            hashlib.sha256(archive.read(f"{ROOT.name}/{name}")).hexdigest() == expected
        )
print(
    f"Verified upload bundle: {len(files)} files, {archive_path.stat().st_size:,} bytes"
)
