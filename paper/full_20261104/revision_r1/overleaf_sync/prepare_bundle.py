"""Package only the current English manuscript's required local sources.

No network, authentication, remote writes, training, or solver calls.
"""
from pathlib import Path
import hashlib
import json
import zipfile


HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
FILES = ('main.tex', 'fig_method.tex', 'refs.bib',
         'elsarticle.cls', 'elsarticle-harv.bst')


def main():
    sources = {}
    for name in FILES:
        path = PAPER / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Missing regular source: {name}')
        sources[name] = path.read_bytes()

    archive = HERE / 'main_overleaf_20261113.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as out:
        for name, data in sources.items():
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 13, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            out.writestr(info, data)

    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        assert bundle.namelist() == list(FILES)
        assert all(bundle.read(name) == data for name, data in sources.items())

    manifest = {
        'prepared_on': '2026-11-13',
        'upload_status': 'not_uploaded',
        'remote_verified': False,
        'project_url': 'https://www.overleaf.com/project/6ad9c20b8e72c0d209b0c86e',
        'main_file': 'main.tex',
        'files': {name: {'bytes': len(data),
                         'sha256': hashlib.sha256(data).hexdigest()}
                  for name, data in sources.items()},
        'archive': archive.name,
        'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
    }
    (HERE / 'bundle_manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    print(f'Prepared {archive.name}: {len(FILES)} files, {archive.stat().st_size} bytes')
    print('Sources are byte-identical to local files. Status: NOT UPLOADED.')


if __name__ == '__main__':
    main()
