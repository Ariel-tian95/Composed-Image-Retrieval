"""Adapt prepared validation data to official Context-I2W paths without edits."""
import argparse
import os
from pathlib import Path
import shutil

def stage(root):
    root=root.resolve();source=root/'data/fashion_iq'
    dest=root/'external/context_i2w/data/fashion-iq'
    for name in ['json','image_splits','images']:(dest/name).mkdir(parents=True,exist_ok=True)
    for cat in ['dress','shirt','toptee']:
        shutil.copy2(source/'captions'/f'cap.{cat}.val.json',dest/'json'/f'cap.{cat}.val.json')
        shutil.copy2(source/'image_splits'/f'split.{cat}.val.json',dest/'image_splits'/f'split.{cat}.val.json')
    count=0
    for image in (source/'images').glob('*.png'):
        target=dest/'images'/image.name
        if not target.exists():
            try:os.link(image,target)
            except OSError:shutil.copy2(image,target)
        count+=1
    print('Official data structure prepared:',count,'images;',dest)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'))
    stage(p.parse_args().root)
