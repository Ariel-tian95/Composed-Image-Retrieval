"""Member B week 10: reproducible data audit and review sheets."""
from pathlib import Path
import argparse, json, hashlib, csv, random
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageOps, ImageDraw, ImageFont

def run(root,out):
    out.mkdir(parents=True,exist_ok=True);data=root/'data/fashion_iq'
    report={};samples=[];required=set()
    for cat in ['dress','shirt','toptee']:
        splits={s:set(json.loads((data/'image_splits'/f'split.{cat}.{s}.json').read_text())) for s in ['train','val','test']}
        rows=json.loads((data/'captions'/f'cap.{cat}.val.json').read_text())
        errors=[]
        for i,r in enumerate(rows):
            if not all(isinstance(r.get(k),str) for k in ['candidate','target']):errors.append([i,'invalid IDs']);continue
            if not isinstance(r.get('captions'),list) or len(r['captions'])!=2 or not all(isinstance(t,str) and t.strip() for t in r['captions']):errors.append([i,'invalid captions'])
            for k in ['candidate','target']:
                if r[k] not in splits['val']:errors.append([i,k+' outside validation gallery'])
            required.update([r['candidate'],r['target']])
        required.update(splits['val'])
        report[cat]={'queries':len(rows),'splits':{s:len(v) for s,v in splits.items()},'overlaps':{a+'_'+b:len(splits[a]&splits[b]) for a,b in [('train','val'),('train','test'),('val','test')]},'annotation_errors':errors}
        picked=random.Random(20261112).sample(list(enumerate(rows)),10)
        sheet=Image.new('RGB',(1200,2500),'white');draw=ImageDraw.Draw(sheet)
        font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16) if Path('C:/Windows/Fonts/arial.ttf').exists() else ImageFont.load_default()
        import textwrap
        for n,(index,r) in enumerate(picked):
            y=n*250
            for j,k in enumerate(['candidate','target']):
                with Image.open(data/'images'/f'{r[k]}.png') as im:
                    tile=ImageOps.contain(im.convert('RGB'),(230,210));sheet.paste(tile,(j*260,y+30))
                draw.text((j*260+5,y+5),k+': '+r[k],fill='black',font=font)
            text='\n'.join(textwrap.wrap(' | '.join(r['captions']),65))
            draw.text((530,y+35),f'{cat} row {index}\n'+text,fill='black',font=font)
            samples.append({'category':cat,'row_index':index,'candidate':r['candidate'],'target':r['target'],'caption_1':r['captions'][0],'caption_2':r['captions'][1],'review_status':'pending_visual_review','notes':''})
        sheet.save(out/f'review_{cat}.jpg',quality=92)
    def inspect(item):
        p=data/'images'/f'{item}.png'
        try:
            raw=p.read_bytes()
            with Image.open(p) as im:im.load();size=im.size;mode=im.mode
            return {'id':item,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'width':size[0],'height':size[1],'mode':mode,'error':''}
        except Exception as e:return {'id':item,'error':str(e)}
    with ThreadPoolExecutor(max_workers=12) as pool:files=list(pool.map(inspect,sorted(required)))
    with (out/'image_manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['id','bytes','sha256','width','height','mode','error']);w.writeheader();w.writerows(files)
    with (out/'review_30.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(samples[0]));w.writeheader();w.writerows(samples)
    result={'categories':report,'unique_required_images':len(required),'file_errors':[r for r in files if r['error']],'visual_review':'pending','scope':'Validation images only; train/test split metadata audited, train/test image completeness not claimed.'}
    (out/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.root,a.out)
