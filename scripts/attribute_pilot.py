"""Frozen-CLIP attribute feasibility baseline. No target-domain model training."""
import argparse,csv,json,hashlib,sys,math,platform,subprocess,datetime
from pathlib import Path

VALUES={'color':['black','white','gray','red','orange','yellow','green','blue','purple','pink','brown','beige','multicolor'],'sleeve_length':['sleeveless','short','elbow','three_quarter','long'],'pattern':['plain','stripe','check','dot','floral','letters_graphics','mixed']}
PHRASES={'sleeveless':'sleeveless','short':'short sleeves','elbow':'elbow-length sleeves','three_quarter':'three-quarter sleeves','long':'long sleeves','plain':'a plain solid pattern','stripe':'a striped pattern','check':'a checked plaid pattern','dot':'a polka dot pattern','floral':'a floral pattern','letters_graphics':'printed lettering or graphics','mixed':'multiple different patterns','multicolor':'multiple colors with no dominant color'}

def read(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

SPECIAL=['unknown','uncertain','not_visible']
CHECKPOINT_SHA='155bb110234bc6aaba60045173eb6bfce402928ad288ba5188dbe13c46676eb4'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def manifest_digest(rows):
    records=sorted((r['image_id'],r['split'],r.get('sha256','')) for r in rows)
    return hashlib.sha256(json.dumps(records,separators=(',',':')).encode()).hexdigest()

def configuration():
    prompts={}
    for group,values in VALUES.items():
        prompts[group]={}
        for value in values:
            phrase=PHRASES.get(value,value)
            texts=[f'a photo of a {value} garment',f'a garment whose main color is {value}'] if group=='color' else [f'a garment with {phrase}',f'a clothing product with {phrase}']
            if value=='multicolor':texts=['a multicolored garment with no dominant color','a garment with multiple equally prominent colors']
            if value=='sleeveless':texts=['a sleeveless garment','a garment without sleeves']
            prompts[group][value]=texts
    return {'values':VALUES,'prompts':prompts,'checkpoint_sha256':CHECKPOINT_SHA,'precision':'float32','text_ensemble':'normalize each prompt; mean; normalize','decision':'argmax cosine; ties follow fixed vocabulary order','abstention':'none; margin is uncalibrated; no threshold selected','preprocess':'official model.clip._transform(input_resolution,is_train=False); RGB','official_commit':'c524d5b9f1551ef14e26791b2bb21654319d0493'}

def protocol_digest():
    return hashlib.sha256(json.dumps(configuration(),sort_keys=True,separators=(',',':')).encode()).hexdigest()

def validate_rows(rows,manifest=None):
    if not rows or len({r['image_id'] for r in rows})!=len(rows):raise ValueError('Empty or duplicate image IDs')
    if any(r['split'] not in ['dev','eval'] for r in rows):raise ValueError('Invalid split')
    for r in rows:
        for g,values in VALUES.items():
            if r.get(g,'').strip() not in values+SPECIAL+['']:raise ValueError(f'Invalid {g} label for {r["image_id"]}')
        if any(r.get(g,'').strip() for g in VALUES) and not r.get('reviewer','').strip():raise ValueError('Annotations require reviewer provenance')
    if manifest is not None:
        reference={r['image_id']:r for r in manifest}
        if set(reference)!={r['image_id'] for r in rows}:raise ValueError('Labels do not match fixed manifest IDs')
        for r in rows:
            original=reference[r['image_id']]
            if r['split']!=original['split'] or r.get('sha256','')!=original.get('sha256',''):raise ValueError('Manifest split/SHA mismatch')

def freeze(manifest,labels,out,reviewer):
    rows=read(labels);base=read(manifest);validate_rows(rows,base)
    dev=[r for r in rows if r['split']=='dev']
    if len(dev)!=50 or sum(r['split']=='eval' for r in base)!=100:raise ValueError('Expected original 50 dev + 100 eval')
    if not reviewer.strip() or any(not r.get('reviewer','').strip() or any(not r.get(g,'').strip() for g in VALUES) for r in dev):raise ValueError('Complete and review all 50 dev rows before freezing')
    if out.exists():raise FileExistsError('Preserve existing lock; choose a new experiment path')
    doc={'schema_version':2,'configuration':configuration(),'protocol_sha256':protocol_digest(),'script_sha256':digest(Path(__file__)),'manifest_sha256':manifest_digest(base),'dev_labels_sha256':digest(labels),'frozen_by':reviewer,'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rule':'Eval labels/results must not guide prompt or threshold choice'}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding='utf-8');print('Protocol frozen:',out)

def verify_lock(lock,rows):
    if lock is None:raise ValueError('Evaluation requires --lock created after dev review')
    doc=json.loads(lock.read_text(encoding='utf-8'))
    if doc['protocol_sha256']!=protocol_digest() or doc['script_sha256']!=digest(Path(__file__)) or doc['manifest_sha256']!=manifest_digest(rows):raise ValueError('Frozen protocol, script or manifest changed')
    return doc

def predict(root,manifest,out,split='dev',lock=None):
    rows_all=read(manifest);validate_rows(rows_all)
    if split=='eval':verify_lock(lock,rows_all)
    if out.exists():raise FileExistsError('Prediction already exists; preserve it or choose a new output')
    import torch
    from PIL import Image
    official=root/'external/context_i2w';sys.path.insert(0,str(official))
    from model.model import build_model
    from model.clip import _transform
    from third_party.open_clip.simple_tokenizer import SimpleTokenizer
    path=root/'checkpoints/context_i2w/official_checkpoint.pt';h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    assert h.hexdigest()==CHECKPOINT_SHA
    official_commit=subprocess.check_output(['git','-C',str(official),'rev-parse','HEAD'],text=True).strip()
    if official_commit!=configuration()['official_commit']:raise ValueError('Official source commit differs from frozen protocol')
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
                prompts=configuration()['prompts'][group][value]
                features.append(normalize(normalize(model.encode_text(tokens(prompts))).mean(0,keepdim=True))[0])
            text_features[group]=torch.stack(features)
        rows=[r for r in rows_all if r['split']==split]
        assert rows,'No samples in requested split'
        for start in range(0,len(rows),8):
            batch=rows[start:start+8];images=[]
            for row in batch:
                image_path=root/'data/fashion_iq/images'/f'{row["image_id"]}.png'
                if row.get('sha256') and digest(image_path)!=row['sha256']:raise ValueError('Original PNG SHA mismatch: '+row['image_id'])
                with Image.open(image_path) as im:images.append(transform(im.convert('RGB')))
            image_features=normalize(model.encode_image(torch.stack(images).to(device)))
            for i,row in enumerate(batch):
                item={'image_id':row['image_id'],'split':row['split'],'attributes':{}}
                for group,values in VALUES.items():
                    scores=(image_features[i]@text_features[group].T).cpu().tolist();order=sorted(range(len(values)),key=lambda k:scores[k],reverse=True)
                    item['attributes'][group]={'prediction':values[order[0]],'score_margin':scores[order[0]]-scores[order[1]],'cosine_scores':dict(zip(values,scores))}
                predictions.append(item)
            print(f'Predicted {min(start+8,len(rows))}/{len(rows)}',flush=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    doc={'schema_version':2,'model':'official Context-I2W checkpoint CLIP encoder; frozen; mapper unused','checkpoint_sha256':CHECKPOINT_SHA,'official_commit':official_commit,'protocol_sha256':protocol_digest(),'manifest_sha256':manifest_digest(rows_all),'script_sha256':digest(Path(__file__)),'configuration':configuration(),'actual_transform':repr(transform),'input_resolution':model.visual.input_resolution,'runtime':{'python':platform.python_version(),'torch':torch.__version__,'device':device,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},'split':split,'scope':'dominant visual color, sleeve length, dominant pattern; no calibrated confidence','predictions':predictions}
    if lock:doc['lock_sha256']=digest(lock)
    out.write_text(json.dumps(doc,indent=2),encoding='utf-8')

def evaluate(labels,predictions,out,split,manifest=None,lock=None):
    all_rows=read(labels);base=read(manifest) if manifest else None;validate_rows(all_rows,base)
    if split=='eval':verify_lock(lock,base or all_rows)
    rows=[r for r in all_rows if r['split']==split]
    if not rows:raise ValueError('No rows for requested split')
    if len({r['image_id'] for r in rows})!=len(rows):raise ValueError('Duplicate label IDs')
    document=json.loads(predictions.read_text(encoding='utf-8'));preds={p['image_id']:p for p in document['predictions']}
    if len(preds)!=len(document['predictions']):raise ValueError('Duplicate prediction IDs')
    if document.get('protocol_sha256')!=protocol_digest():raise ValueError('Prediction protocol mismatch or missing provenance; rerun with current script')
    if document.get('script_sha256')!=digest(Path(__file__)):raise ValueError('Prediction script changed; rerun development stage before freezing')
    if base is not None and document.get('manifest_sha256')!=manifest_digest(base):raise ValueError('Prediction manifest mismatch')
    if set(preds)!={r['image_id'] for r in rows}:raise ValueError('Prediction IDs must exactly match requested split')
    if split=='eval' and document.get('lock_sha256')!=digest(lock):raise ValueError('Prediction lock differs')
    for row in rows:
        p=preds[row['image_id']]
        if p['split']!=split:raise ValueError('Prediction/label split mismatch')
        for group,values in VALUES.items():
            a=p['attributes'][group];scores=a.get('cosine_scores',{})
            if set(scores)!=set(values) or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in scores.values()):raise ValueError('Missing/nonfinite/invalid cosine scores')
            ordered=sorted(values,key=lambda v:scores[v],reverse=True)
            if a['prediction']!=ordered[0] or not math.isclose(a['score_margin'],scores[ordered[0]]-scores[ordered[1]],abs_tol=1e-6):raise ValueError('Prediction or margin inconsistent with scores')
    result={'split':split,'sample_count':len(rows),'attributes':{},'protocol_sha256':protocol_digest(),'prediction_file_sha256':digest(predictions),'label_file_sha256':digest(labels),'note':'Macro F1 averages classes with ground-truth support; absent classes listed. Margin is not calibrated probability. Unknown/not_visible/uncertain excluded from scored truth.'}
    errors=[]
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
        excluded=collections_counter(r.get(group,'').strip() or 'blank' for r in rows if r.get(group,'').strip() not in values)
        if not known:result['attributes'][group]={'status':'pending_labels','known':0,'label_coverage':0,'excluded':excluded,'accuracy':None,'macro_f1':None};continue
        per_class={}
        for label in values:
            tp=sum(a==label and b==label for a,b in known);fp=sum(a!=label and b==label for a,b in known);fn=sum(a==label and b!=label for a,b in known)
            per_class[label]={'support':tp+fn,'predicted':tp+fp,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}
        confusion={label:{other:sum(a==label and b==other for a,b in known) for other in values} for label in values}
        supported=[v['f1'] for v in per_class.values() if v['support']>0]
        result['attributes'][group]={'status':'evaluated','known':len(known),'label_coverage':len(known)/len(rows),'excluded':excluded,'accuracy':sum(a==b for a,b in known)/len(known),'macro_f1':sum(supported)/len(supported),'macro_f1_all_fixed_classes_zero_for_absent':sum(v['f1'] or 0 for v in per_class.values())/len(values),'missing_truth_classes':[k for k,v in per_class.items() if not v['support']],'per_class':per_class,'confusion':confusion}
        for row in rows:
            truth=row.get(group,'').strip();prediction=preds[row['image_id']]['attributes'][group]
            if truth in values and truth!=prediction['prediction']:errors.append({'image_id':row['image_id'],'split':split,'attribute':group,'truth':truth,'prediction':prediction['prediction'],'margin':prediction['score_margin'],'reviewer':row['reviewer'],'notes':row.get('notes','')})
    result['annotation_complete']=all(r.get('reviewer','').strip() and all(r.get(g,'').strip() for g in VALUES) for r in rows)
    result['status']='evaluated' if all(v['status']=='evaluated' for v in result['attributes'].values()) else 'pending_labels'
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError('Preserve earlier metrics; choose a new output')
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    error_path=out.with_name(out.stem+'_errors.csv')
    with error_path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['image_id','split','attribute','truth','prediction','margin','reviewer','notes']);writer.writeheader();writer.writerows(errors)
    lines=['# Attribute reliability evaluation',f'Split: {split}. Status: {result["status"]}.',f'Annotation complete: {result["annotation_complete"]}.','No automatic acceptance threshold; review per-class support and errors.','']
    for group,a in result['attributes'].items():lines.append(f'{group}: known={a["known"]}/{len(rows)}, accuracy={a.get("accuracy")}, macro F1={a.get("macro_f1")}, excluded={a["excluded"]}')
    lines+=['','Without independent labels and adequate per-class support, no reliability claim is made.','No confidence rejection threshold was selected. Retention of scored known labels is 100%; scores are not calibrated probabilities.','This target-domain diagnostic is not an external benchmark; three dominant attributes do not validate all 25 mappings or feedback performance.']
    out.with_suffix('.md').write_text('\n'.join(lines),encoding='utf-8');print(f'{split}: {result["status"]}; {len(rows)} rows; error cases: {len(errors)}')

def collections_counter(items):
    from collections import Counter
    return dict(Counter(items))

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('predict');a.add_argument('--root',type=Path,required=True);a.add_argument('--manifest',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--split',choices=['dev','eval'],default='dev')
    a.add_argument('--lock',type=Path)
    a=sub.add_parser('evaluate');a.add_argument('--labels',type=Path,required=True);a.add_argument('--predictions',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--split',choices=['dev','eval'],default='dev')
    a.add_argument('--manifest',type=Path);a.add_argument('--lock',type=Path)
    a=sub.add_parser('freeze');a.add_argument('--manifest',type=Path,required=True);a.add_argument('--labels',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--reviewer',required=True)
    args=vars(p.parse_args());command=args.pop('command');{'predict':predict,'evaluate':evaluate,'freeze':freeze}[command](**args)
