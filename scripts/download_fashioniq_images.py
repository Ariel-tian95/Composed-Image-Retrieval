"""Fetch official FashionIQ validation images from the published URL mapping.

Converts decoded RGB pixels to PNG, preserving official product IDs.
Failures are recorded; incomplete datasets must not be reported as full baseline.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
from pathlib import Path
import urllib.request
from PIL import Image

def download(root, limit=0, workers=12):
    data=root/'data/fashion_iq'; mapping={}; required=set()
    for cat in ['dress','shirt','toptee']:
        for line in (root/'external/fashion_iq_metadata/image_url'/f'asin2url.{cat}.txt').read_text().splitlines():
            parts=line.split()
            if len(parts)==2:mapping[parts[0]]=parts[1].replace('http://','https://',1)
        required.update(json.loads((data/'image_splits'/f'split.{cat}.val.json').read_text()))
        for row in json.loads((data/'captions'/f'cap.{cat}.val.json').read_text()):
            required.update([row['candidate'],row['target']])
    ids=sorted(required)
    if limit:ids=ids[:limit]
    (data/'images').mkdir(parents=True,exist_ok=True)
    def one(item):
        dest=data/'images'/f'{item}.png'
        if dest.exists():
            try:
                with Image.open(dest) as im:im.verify()
                return item,None
            except Exception:pass
        if item not in mapping:return item,'No URL in official mapping'
        errors=[]
        # Same Amazon image path on its current CDN is a fallback when the legacy host fails.
        original=mapping[item]
        urls=[original,original.replace('ecx.images-amazon.com','m.media-amazon.com')]
        for url in dict.fromkeys(urls):
            try:
                request=urllib.request.Request(url,headers={'User-Agent':'FashionIQ research data preparation'})
                with urllib.request.urlopen(request,timeout=20) as r:raw=r.read()
                with Image.open(io.BytesIO(raw)) as image:
                    image.convert('RGB').save(dest.with_suffix('.partial'),format='PNG')
                dest.with_suffix('.partial').replace(dest)
                return item,None
            except Exception as e:errors.append(str(e))
        return item,'; '.join(errors)
    failures={}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(one,i) for i in ids]
        for n,f in enumerate(as_completed(futures),1):
            item,error=f.result()
            if error:failures[item]=error
            if n%100==0:print(f'Checked {n}/{len(ids)}, failed {len(failures)}',flush=True)
    result={'requested':len(ids),'success':len(ids)-len(failures),'failures':failures,'sample_only':bool(limit),'image_processing':'Decode source JPEG, RGB conversion, save lossless PNG; no resizing.'}
    (root/'results').mkdir(parents=True,exist_ok=True)
    (root/'results/image_download.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='failures'},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--limit',type=int,default=0);p.add_argument('--workers',type=int,default=12)
    a=p.parse_args();download(a.root,a.limit,a.workers)
