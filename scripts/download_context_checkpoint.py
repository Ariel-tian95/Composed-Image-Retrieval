"""Download official checkpoint. A downloaded checkpoint is not a loaded model."""
import argparse
from pathlib import Path
import hashlib
import json

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path('.'))
    root=p.parse_args().root
    import gdown
    dest=root/'checkpoints/context_i2w/official_checkpoint.pt'
    expected_bytes=2532409369
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and dest.stat().st_size!=expected_bytes:
        raise RuntimeError('Existing checkpoint has an unexpected size; keep it separate and retry the download.')
    if not dest.exists():
        partial=dest.with_suffix('.partial')
        result=gdown.download(id='1yHv9SgY1-2KrAvd2MmFqH40x4P3gLotQ',output=str(partial),resume=True)
        if not result:
            raise RuntimeError('Download incomplete')
        if partial.stat().st_size!=expected_bytes:
            raise RuntimeError('Official checkpoint size differs from the verified source; inspect before using.')
        partial.replace(dest)
    sha=hashlib.sha256()
    with dest.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            sha.update(block)
    record={'source':'https://drive.google.com/file/d/1yHv9SgY1-2KrAvd2MmFqH40x4P3gLotQ/view','bytes':dest.stat().st_size,'sha256':sha.hexdigest(),'model_load_tested':False}
    dest.with_suffix('.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))
