"""One-time, checksum-verified transfer of the tested source into this repository."""
import base64
import hashlib
import json
import lzma
from pathlib import Path, PurePosixPath
import shutil
import tempfile

root = Path.cwd().resolve()
delivery = root / '.moka-delivery'
manifest = json.loads((delivery / 'manifest.json').read_text('utf-8'))
if manifest.get('schema') != 'moka-source-delivery/1':
    raise SystemExit('Unknown delivery schema')
parts = []
for name, expected in sorted(manifest['parts'].items()):
    if not name.startswith('part-') or '/' in name or '\\' in name:
        raise SystemExit('Invalid delivery part')
    data = (delivery / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise SystemExit('Delivery checksum mismatch: ' + name)
    parts.append(data)
archive = base64.b64decode(b''.join(parts), validate=True)
if hashlib.sha256(archive).hexdigest() != manifest['archive_sha256']:
    raise SystemExit('Archive checksum mismatch')
raw = lzma.LZMADecompressor(memlimit=256_000_000).decompress(archive, max_length=5_000_000)
files = json.loads(raw)
if set(files) != set(manifest['files']) or len(files) > 256:
    raise SystemExit('Source manifest mismatch')
with tempfile.TemporaryDirectory(prefix='moka-source-') as folder:
    staged = Path(folder)
    for name, content in files.items():
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or '\\' in name or '.git' in path.parts:
            raise SystemExit('Unsafe source path')
        data = content.encode('utf-8')
        if hashlib.sha256(data).hexdigest() != manifest['files'][name]:
            raise SystemExit('Source checksum mismatch: ' + name)
        target = staged / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        existing = root / name
        if existing.exists() and existing.read_bytes() != data:
            raise SystemExit('Refusing to replace an independently changed file: ' + name)
    for name in files:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(staged / name, target)
        if name in manifest.get('executable', []):
            target.chmod(0o755)
shutil.rmtree(delivery)
print('Materialized', len(files), 'verified source files; temporary delivery removed.')
print('Archive SHA256:', manifest['archive_sha256'])
