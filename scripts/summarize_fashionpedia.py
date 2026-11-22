"""Export original attribute IDs and validate global attribute references."""
import argparse
import csv
import json
from pathlib import Path

def summarize(root):
    stats={}; attributes=None; image_sets={}
    for split in ['train','val']:
        path=root/'data/fashionpedia'/f'attributes_{split}2020.json'
        data=json.loads(path.read_text(encoding='utf-8'))
        labels={a['id'] for a in data['attributes']}
        images={i['id'] for i in data['images']}
        unknown_labels=set();unknown_images=set()
        for row in data['annotations']:
            unknown_labels.update(set(row['attribute_ids'])-labels)
            if row['image_id'] not in images:unknown_images.add(row['image_id'])
        stats[split]={'images':len(images),'duplicate_image_ids':len(data['images'])-len(images),'annotations':len(data['annotations']),'attributes':len(labels),'unknown_attribute_ids':sorted(unknown_labels),'unknown_image_ids':sorted(unknown_images)}
        image_sets[split]=images
        if attributes is None:attributes=data['attributes']
        elif attributes!=data['attributes']:raise ValueError('Train/val attribute definitions differ')
    stats['train_val_overlap']=len(image_sets['train']&image_sets['val'])
    dest=root/'results';dest.mkdir(parents=True,exist_ok=True)
    with (dest/'fashionpedia_attribute_inventory.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['id','name','supercategory']);writer.writeheader()
        writer.writerows({k:a.get(k,'') for k in writer.fieldnames} for a in attributes)
    (dest/'fashionpedia_quality.json').write_text(json.dumps(stats,indent=2))
    print(json.dumps(stats,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'))
    summarize(p.parse_args().root)
