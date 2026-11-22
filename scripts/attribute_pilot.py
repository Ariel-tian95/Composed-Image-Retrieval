"""Frozen-CLIP attribute feasibility baseline. No target-domain model training."""
import argparse,csv,json,hashlib,sys
from pathlib import Path

VALUES={'color':['black','white','gray','red','orange','yellow','green','blue','purple','pink','brown','beige','multicolor'],'sleeve_length':['sleeveless','short','elbow','three_quarter','long'],'pattern':['plain','stripe','check','dot','floral','letters_graphics','mixed']}
PHRASES={'sleeveless':'sleeveless','short':'short sleeves','elbow':'elbow-length sleeves','three_quarter':'three-quarter sleeves','long':'long sleeves','plain':'a plain solid pattern','stripe':'a striped pattern','check':'a checked plaid pattern','dot':'a polka dot pattern','floral':'a floral pattern','letters_graphics':'printed lettering or graphics','mixed':'multiple different patterns','multicolor':'multiple colors with no dominant color'}

def read(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def predict(root,manifest,out,split='dev'):
    import torch
    from PIL import Image
    official=root/'external/context_i2w';sys.path.insert(0,str(official))
    from model.model import build_model
    from model.clip import _transform
    from third_party.open_clip.simple_tokenizer import SimpleTokenizer
    path=root/'checkpoints/context_i2w/official_checkpoint.pt';h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    assert h.hexdigest()=='155bb110234bc6aaba60045173eb6bfce402928ad288ba5188dbe13c46676eb4'
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    model=build_model({k.removeprefix('module.'):v for k,v in checkpoint['state_dict'].items()}).float();del checkpoint
    device='cuda' if torch.cuda.is_available() else 'cpu';model=model.to(device).eval()
    transform=_transform(model.visual.input_resolution,is_train=False);tokenizer=SimpleTokenizer()
    def tokens(sentences):
        result=torch.zeros(len(sentences),77,dtype=torch.long)
        for i,text in enumerate(sentences):
            t=[tokenizer.encoder['<|startoftext|>']]+tokenizer.encode(text)+[tokenizer.encoder['<|endoftext|>']]
            assert len(t)<=77;result[i,:len(t)]=torch.tensor(t)
        return result.to(device)
    normalize=lambda x:torch.nn.functional.normalize(x.float(),dim=-1)
    predictions=[]
    with torch.inference_mode():
        text_features={}
        for group,values in VALUES.items():
            features=[]
            for value in values:
                phrase=PHRASES.get(value,value)
                prompts=[f'a photo of a {value} garment',f'a garment whose main color is {value}'] if group=='color' else [f'a garment with {phrase}',f'a clothing product with {phrase}']
                if value=='multicolor':prompts=['a multicolored garment with no dominant color','a garment with multiple equally prominent colors']
                if value=='sleeveless':prompts=['a sleeveless garment','a garment without sleeves']
                features.append(normalize(normalize(model.encode_text(tokens(prompts))).mean(0,keepdim=True))[0])
            text_features[group]=torch.stack(features)
        rows=[r for r in read(manifest) if r['split']==split]
        assert rows,'No samples in requested split'
        for start in range(0,len(rows),8):
            batch=rows[start:start+8];images=[]
            for row in batch:
                with Image.open(root/'data/fashion_iq/images'/f'{row["image_id"]}.png') as im:images.append(transform(im.convert('RGB')))
            image_features=normalize(model.encode_image(torch.stack(images).to(device)))
            for i,row in enumerate(batch):
                item={'image_id':row['image_id'],'split':row['split'],'attributes':{}}
                for group,values in VALUES.items():
                    scores=(image_features[i]@text_features[group].T).cpu().tolist();order=sorted(range(len(values)),key=lambda k:scores[k],reverse=True)
                    item['attributes'][group]={'prediction':values[order[0]],'score_margin':scores[order[0]]-scores[order[1]],'cosine_scores':dict(zip(values,scores))}
                predictions.append(item)
            print(f'Predicted {min(start+8,len(rows))}/{len(rows)}',flush=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'model':'official Context-I2W checkpoint CLIP encoder; frozen; mapper unused','device':device,'scope':'dominant visual color, sleeve length, dominant pattern; no calibrated confidence','predictions':predictions},indent=2))

def evaluate(labels,predictions,out,split):
    rows=[r for r in read(labels) if r['split']==split]
    if len({r['image_id'] for r in rows})!=len(rows):raise ValueError('Duplicate label IDs')
    document=json.loads(predictions.read_text());preds={p['image_id']:p for p in document['predictions']}
    if len(preds)!=len(document['predictions']):raise ValueError('Duplicate prediction IDs')
    result={'split':split,'sample_count':len(rows),'attributes':{},'note':'Margin is not calibrated probability. Unknown/not_visible/uncertain are not negatives.'}
    for group,values in VALUES.items():
        known=[]
        for row in rows:
            truth=row[group].strip()
            if truth not in values+['','unknown','uncertain','not_visible']:raise ValueError(f'Invalid {group} label: {truth}')
            if truth in values:
                if not row.get('reviewer','').strip():raise ValueError('Known labels require reviewer provenance')
                if row['image_id'] not in preds:raise ValueError('Missing prediction '+row['image_id'])
                if preds[row['image_id']].get('split')!=split:raise ValueError('Prediction/label split mismatch')
                known.append((truth,preds[row['image_id']]['attributes'][group]['prediction']))
        if not known:result['attributes'][group]={'status':'pending_labels','known':0};continue
        classes=sorted({a for a,b in known}|{b for a,b in known});per_class={}
        for label in classes:
            tp=sum(a==label and b==label for a,b in known);fp=sum(a!=label and b==label for a,b in known);fn=sum(a==label and b!=label for a,b in known)
            per_class[label]={'support':tp+fn,'precision':tp/(tp+fp) if tp+fp else 0,'recall':tp/(tp+fn) if tp+fn else 0,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0}
        confusion={label:{other:sum(a==label and b==other for a,b in known) for other in values} for label in values}
        result['attributes'][group]={'status':'evaluated','known':len(known),'label_coverage':len(known)/len(rows),'accuracy':sum(a==b for a,b in known)/len(known),'macro_f1':sum(v['f1'] for v in per_class.values())/len(per_class),'per_class':per_class,'confusion':confusion}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('predict');a.add_argument('--root',type=Path,required=True);a.add_argument('--manifest',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--split',choices=['dev','eval'],default='dev')
    a=sub.add_parser('evaluate');a.add_argument('--labels',type=Path,required=True);a.add_argument('--predictions',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--split',choices=['dev','eval'],default='dev')
    args=vars(p.parse_args());command=args.pop('command');(predict if command=='predict' else evaluate)(**args)
