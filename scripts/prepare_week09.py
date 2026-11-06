"""Member B: reproducible source and annotation preparation. Python 3.10+."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

SOURCES = {
    'context_i2w': 'https://github.com/Pter61/context-i2w.git',
    'fashion_iq': 'https://github.com/XiaoxiaoGuo/fashion-iq.git',
    'fashion_iq_metadata': 'https://github.com/hongwang600/fashion-iq-metadata.git',
}

def prepare(root):
    root = root.resolve()
    records = {}
    for name, url in SOURCES.items():
        dest = root / 'external' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            subprocess.run(['git', 'clone', '--depth', '1', url, str(dest)], check=True)
        commit = subprocess.check_output(['git','-C',str(dest),'rev-parse','HEAD'],text=True).strip()
        records[name] = {'url':url,'commit':commit}
    import shutil
    data = root/'data/fashion_iq'
    for name in ['captions','image_splits']:
        shutil.copytree(root/'external/fashion_iq'/name,data/name,dirs_exist_ok=True)
    (data/'images').mkdir(parents=True,exist_ok=True)
    for name in ['attributes_train2020.json','attributes_val2020.json']:
        dest = root/'data/fashionpedia'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        valid = False
        if dest.exists():
            try:
                with dest.open(encoding='utf-8') as f: json.load(f)
                valid = True
            except (ValueError, OSError): pass
        if not valid:
            tmp = dest.with_suffix('.partial')
            urllib.request.urlretrieve('https://s3.amazonaws.com/ifashionist-dataset/annotations/'+name,tmp)
            with tmp.open(encoding='utf-8') as f:
                json.load(f)
            tmp.replace(dest)
        records[name]={'bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
    (root/'checkpoints/context_i2w').mkdir(parents=True,exist_ok=True)
    (root/'results').mkdir(parents=True,exist_ok=True)
    (root/'results/source_manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print('Prepared sources and annotations:',root)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('.'))
    prepare(parser.parse_args().root)
