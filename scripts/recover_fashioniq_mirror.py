"""Recover missing validation IDs from the WISER authors' published mirror.

Requires 7z on PATH, or the py7zr package. Keeps original FashionIQ annotations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import tarfile
import urllib.request
from PIL import Image

URL='https://huggingface.co/datasets/Physicsmile/WISER/resolve/main/fashion-iq.7z'
SHA256='06b4542d89b5e3b040017cd75e822236e094a110bde6a05ab06986e79d1ab1a8'

def recover(root, archive=None):
    root=root.resolve()
    if archive is None:
        archive=root/'downloads/fashion-iq-wiser.7z'
    archive=archive.resolve();archive.parent.mkdir(parents=True,exist_ok=True)
    if not archive.exists():
        temp=archive.with_suffix('.partial')
        urllib.request.urlretrieve(URL,temp)
        temp.replace(archive)
    sha=hashlib.sha256()
    with archive.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):sha.update(b)
    if sha.hexdigest()!=SHA256:raise ValueError('Mirror SHA256 does not match publisher')
    required=set()
    for cat in ['dress','shirt','toptee']:
        required.update(json.loads((root/'data/fashion_iq/image_splits'/f'split.{cat}.val.json').read_text()))
        for row in json.loads((root/'data/fashion_iq/captions'/f'cap.{cat}.val.json').read_text()):
            required.update([row['candidate'],row['target']])
    images=root/'data/fashion_iq/images';images.mkdir(parents=True,exist_ok=True)
    missing={i for i in required if not (images/f'{i}.png').exists()}
    executable=shutil.which('7z')
    if executable:
        listing=subprocess.check_output([executable,'l','-slt',str(archive)],text=True,encoding='utf-8',errors='replace')
        names=[line[7:] for line in listing.splitlines() if line.startswith('Path = ')]
    else:
        import py7zr
        with py7zr.SevenZipFile(archive) as z:names=z.getnames()
    targets=[n for n in names if Path(n.replace('\\','/')).stem in missing and Path(n).suffix.lower() in ['.png','.jpg','.jpeg']]
    nested=[n for n in names if Path(n.replace('\\','/')).name=='images.tar.gz']
    for name in targets:
        p=Path(name.replace('\\','/'))
        if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe archive path')
    (root/'results').mkdir(exist_ok=True)
    recovered=[]
    with tempfile.TemporaryDirectory(prefix='fashioniq-recovery-',dir=root) as temp:
        temp=Path(temp)
        extract_targets=targets or nested
        if executable and extract_targets:
            filelist=temp/'targets.txt';filelist.write_text('\n'.join(extract_targets),encoding='utf-8')
            subprocess.run([executable,'x',str(archive),f'-o{temp}',f'-i@{filelist}','-y','-scsUTF-8'],check=True,stdout=subprocess.DEVNULL)
        elif extract_targets:
            with py7zr.SevenZipFile(archive) as z:z.extract(temp,targets=extract_targets)
        for name in nested if not targets else []:
            with tarfile.open(temp/Path(name.replace('\\','/')),mode='r|gz') as tar:
                for member in tar:
                    item=Path(member.name).stem
                    if not member.isfile() or item not in missing or item in recovered:continue
                    f=tar.extractfile(member)
                    if f is None:continue
                    with f, Image.open(f) as image:image.convert('RGB').save(images/f'{item}.png')
                    recovered.append(item)
        for name in targets:
            source=temp/Path(name.replace('\\','/'))
            item=source.stem
            if item in recovered:continue
            with Image.open(source) as image:image.convert('RGB').save(images/f'{item}.png')
            recovered.append(item)
    record={'mirror_url':URL,'sha256':sha.hexdigest(),'recovered_ids':sorted(recovered),'remaining_missing_ids':sorted(missing-set(recovered)),'conversion':'RGB PNG; no resizing'}
    (root/'results/mirror_recovery.json').write_text(json.dumps(record,indent=2))
    print('Recovered:',len(recovered),'remaining:',len(record['remaining_missing_ids']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--archive',type=Path)
    a=p.parse_args();recover(a.root,a.archive)
