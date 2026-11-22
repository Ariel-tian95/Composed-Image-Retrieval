"""Audit official validation annotations and image availability; never skip silently."""
import argparse
import json
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

def audit(root):
    data=root/'data/fashion_iq'
    report={'split':'val','categories':{},'all_images_present':True}
    all_ids=set()
    for cat in ['dress','shirt','toptee']:
        rows=json.loads((data/'captions'/f'cap.{cat}.val.json').read_text())
        gallery=json.loads((data/'image_splits'/f'split.{cat}.val.json').read_text())
        gallery_ids=set(gallery)
        errors=[]
        required=set(gallery)
        for i,row in enumerate(rows):
            if not all(k in row for k in ['candidate','target','captions']):
                errors.append(f'row {i}: missing fields'); continue
            required.update([row['candidate'],row['target']])
            if len(row['captions'])!=2 or not all(isinstance(c,str) for c in row['captions']):
                errors.append(f'row {i}: malformed captions')
            if row['target'] not in gallery_ids:
                errors.append(f'row {i}: target outside gallery')
        missing=sorted(i for i in required if not (data/'images'/f'{i}.png').is_file())
        def verify(item):
            try:
                with Image.open(data/'images'/f'{item}.png') as image: image.verify()
                return None
            except Exception: return item
        with ThreadPoolExecutor(max_workers=12) as pool:
            corrupt=sorted(item for item in pool.map(verify,sorted(required-set(missing))) if item)
        all_ids.update(required)
        report['categories'][cat]={'queries':len(rows),'gallery':len(gallery),'duplicate_gallery_ids':len(gallery)-len(gallery_ids),'required':len(required),'missing':len(missing),'corrupt':corrupt,'annotation_errors':errors}
        (root/'results').mkdir(parents=True,exist_ok=True)
        (root/'results'/f'missing_{cat}.txt').write_text('\n'.join(missing))
        report['all_images_present'] &= not missing and not corrupt and not errors
        print(f'{cat}: {len(rows)} queries, {len(missing)} missing, {len(corrupt)} corrupt',flush=True)
    report['unique_required_images']=len(all_ids)
    (root/'results/fashioniq_quality.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path('.'))
    result=audit(p.parse_args().root)
    raise SystemExit(0 if result['all_images_present'] else 2)
