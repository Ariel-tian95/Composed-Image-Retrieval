"""Context-I2W inference using official architecture, checkpoint and query format.
Standalone runner avoids importing unused training dependencies. GPU execution pending.
"""
import argparse, hashlib, json, sys, os
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--category',choices=['dress','shirt','toptee'],default='dress');p.add_argument('--queries',type=int,default=1);p.add_argument('--gallery-limit',type=int,default=0);p.add_argument('--batch-size',type=int,default=16);a=p.parse_args()
    root=a.root.resolve();official=root/'external/context_i2w';sys.path.insert(0,str(official))
    import torch
    from PIL import Image
    from model.model import build_model,IM2TEXT_OWM_Attn
    from model.clip import _transform
    from third_party.open_clip.simple_tokenizer import SimpleTokenizer
    assert torch.cuda.is_available(),'GPU required; enable Kaggle accelerator'
    checkpoint=root/'checkpoints/context_i2w/official_checkpoint.pt'
    sha=hashlib.sha256()
    with checkpoint.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):sha.update(b)
    assert sha.hexdigest()=='155bb110234bc6aaba60045173eb6bfce402928ad288ba5188dbe13c46676eb4','Checkpoint hash differs from verified official file'
    # Scoped to this exact verified official checkpoint; no global torch.load override.
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
    def strip(sd):return {k.removeprefix('module.'):v for k,v in sd.items()}
    sd=strip(ckpt['state_dict']);mapping=strip(ckpt['state_dict_img2text'])
    model=build_model(sd).float()
    mapper=IM2TEXT_OWM_Attn(transformer_width=model.transformer_width,vision_width=model.vision_width,embed_dim=model.embed_dim,middle_dim=512,output_dim=model.token_embedding.weight.shape[1],n_layer=2)
    mapper.load_state_dict(mapping,strict=True);del ckpt,sd,mapping
    model=model.cuda().eval();mapper=mapper.float().cuda().eval()
    print('PASS: official model and mapping weights loaded strictly',flush=True)
    transform=_transform(model.visual.input_resolution,is_train=False);tok=SimpleTokenizer()
    def tokenize(text):
        ids=[tok.encoder['<|startoftext|>']]+tok.encode(text)+[tok.encoder['<|endoftext|>']]
        if len(ids)>77:raise ValueError('Caption exceeds official context length')
        t=torch.zeros(1,77,dtype=torch.long,device='cuda');t[0,:len(ids)]=torch.tensor(ids,device='cuda');return t
    data=root/'data/fashion_iq';rows=json.loads((data/'captions'/f'cap.{a.category}.val.json').read_text());gallery=json.loads((data/'image_splits'/f'split.{a.category}.val.json').read_text())
    expected={'dress':2017,'shirt':2038,'toptee':1961};assert len(rows)==expected[a.category]
    if a.queries:rows=rows[:a.queries]
    if a.gallery_limit:
        selected=set(gallery[:a.gallery_limit]);selected.update(r['target'] for r in rows);gallery=[i for i in gallery if i in selected]
    def image(i):
        with Image.open(data/'images'/f'{i}.png') as im:return transform(im)
    features=[]
    with torch.inference_mode():
        for start in range(0,len(gallery),a.batch_size):
            batch=torch.stack([image(i) for i in gallery[start:start+a.batch_size]]).cuda()
            f=model.encode_image(batch);features.append(torch.nn.functional.normalize(f.float(),dim=-1).cpu())
            if start%320==0:print('Gallery:',start,'/',len(gallery),flush=True)
        features=torch.cat(features).cuda();predictions=[];hits={10:0,50:0}
        split=tok.encode('<|replace|>')[0]
        for n,r in enumerate(rows):
            # Matches official FashionIQ.return_all: caption 2 then caption 1.
            text=tokenize('a photo of <|replace|> , {} and {}'.format(r['captions'][1],r['captions'][0]))
            ref=image(r['candidate']).unsqueeze(0).cuda()
            pseudo=mapper(model.encode_text(text),model.encode_image_tokens(ref))
            query=model.encode_text_img_retrieval(text,pseudo,split_ind=split,repeat=False)
            query=torch.nn.functional.normalize(query.float(),dim=-1)
            order=(query@features.T)[0].argsort(descending=True).cpu().tolist()
            rank=order.index(gallery.index(r['target']))+1
            for k in hits:hits[k]+=rank<=k
            predictions.append({'candidate':r['candidate'],'target':r['target'],'captions':r['captions'],'target_rank':rank,'top10':[gallery[i] for i in order[:10]]})
            if n%100==0:print('Queries:',n,'/',len(rows),flush=True)
    result={'category':a.category,'queries':len(rows),'gallery':len(gallery),'full_validation':not a.queries and not a.gallery_limit,'protocol':'Official caption order and candidate pool; reference retained as in official get_metrics_fashion; float32 numerical execution.','R@10':100*hits[10]/len(rows),'R@50':100*hits[50]/len(rows),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'predictions':predictions}
    out=root/'results/baseline';out.mkdir(parents=True,exist_ok=True);name=f'{a.category}_'+('full' if result['full_validation'] else 'smoke')+'.json';(out/name).write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='predictions'},indent=2),flush=True)

if __name__=='__main__':main()
