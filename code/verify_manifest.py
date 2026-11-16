"""Verify packaged file hashes with Python's standard library."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'MANIFEST_SHA256.json').read_text('utf-8'))
    failures = []
    for name, expected in manifest['files'].items():
        path = ROOT / name
        if not path.is_file():
            failures.append(name + ': missing')
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(name + ': changed')
    if failures:
        raise SystemExit('\n'.join(failures))
    print(f"Verified {len(manifest['files'])} packaged files. Generated recomputed/ outputs are not part of the manifest.")


if __name__ == '__main__':
    main()
