"""Unpack the read-only dashboard assets on first server launch."""
from pathlib import Path
import hashlib, tempfile, zipfile, shutil, sys, runpy
base = Path(__file__).resolve().parent
archive = base / 'dashboard_bundle.zip'
version = hashlib.sha256(archive.read_bytes()).hexdigest()[:16]
work = Path(tempfile.gettempdir()) / ('image-harness-' + version)
if not (work / '.ready').exists():
    work.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            if not (work / entry.filename).resolve().is_relative_to(work.resolve()):
                raise ValueError('Invalid bundled path')
        z.extractall(work)
    for kind in ['raw', 'harness']:
        shutil.copyfile(base / (kind + '_model.keras'), work / 'part3_harness/results/current' / (kind + '_model.keras'))
    (work / '.ready').touch()
sys.path.insert(0, str(work))
runpy.run_path(str(base / 'dashboard.py'), run_name='__main__')
