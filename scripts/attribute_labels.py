"""Merge independently exported human CSVs without overwriting disagreements."""
import argparse,csv,json
from pathlib import Path
from attribute_pilot import read,validate_rows,VALUES

def merge(manifest,inputs,out):
    if out.exists():raise FileExistsError('Choose a new output; original annotations are preserved')
    base=read(manifest);validate_rows(base);lookup={r['image_id']:dict(r) for r in base}
    for r in lookup.values():
        for field in [*VALUES,'reviewer','notes']:r[field]=''
    conflicts=[]
    for path in inputs:
        incoming=read(path);validate_rows(incoming)
        for r in incoming:
            if r['image_id'] not in lookup:raise ValueError('Unknown image ID')
            dest=lookup[r['image_id']]
            if r['split']!=dest['split'] or r.get('sha256','')!=dest.get('sha256',''):raise ValueError('Split/SHA changed')
            annotated=any(r.get(g,'').strip() for g in VALUES)
            if not annotated:continue
            if any(dest.get(g,'').strip() for g in VALUES):
                # Different reviewers or opinions require an explicit adjudicated CSV.
                if any(dest.get(k,'')!=r.get(k,'') for k in [*VALUES,'reviewer','notes']):
                    conflicts.append({'image_id':r['image_id'],'first':dict(dest),'second':r,'second_file':str(path)})
                continue
            for k in [*VALUES,'reviewer','notes']:dest[k]=r.get(k,'')
    out.parent.mkdir(parents=True,exist_ok=True)
    if conflicts:
        out.with_suffix('.conflicts.json').write_text(json.dumps(conflicts,ensure_ascii=False,indent=2),encoding='utf-8')
        raise ValueError(f'{len(conflicts)} disagreements; no merged truth written. Resolve explicitly and preserve originals.')
    with out.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(base[0]));w.writeheader();w.writerows(lookup.values())
    print('Merged 150 fixed rows; original files unchanged:',out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--inputs',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    merge(**vars(p.parse_args()))
