from argparse import ArgumentParser
from glob import glob
import json
import os
import re

from PIL import Image
from tqdm import tqdm

if __name__=="__main__":
    parser = ArgumentParser()

    parser.add_argument("--danbooru-dir", help="Danbooru SFW 512px directory.")
    parser.add_argument("--dataset-fn", help="Dataset JSON file.")
    parser.add_argument("--save-dir", help="Directory to save dataset images.")

    args = parser.parse_args()

    SIZE = 224

    with open(args.dataset_fn) as h:
        data = json.load(h)

    print("Loading list of Danbooru 2020 iamges.")
    id2fn = {}
    for fn in tqdm(glob(os.path.join(args.danbooru_dir, "**"), recursive=True)):
        if os.path.isfile(fn):
            m = re.search(r"(?P<id>[0-9]+).jpg", fn)
            if m is not None:
                _id = int(m.group("id"))
                id2fn[_id] = fn
    
    print("Cropping images.")
    for character_name, infos in tqdm(data.items()):
        os.makedirs(os.path.join(args.save_dir, character_name), exist_ok=True)

        for info in infos:
            fn = id2fn[info["id"]]
            img = Image.open(fn)
            img = img.crop(info["bbox"])
            img = img.resize((SIZE, SIZE))

            img.save(os.path.join(args.save_dir, character_name, os.path.basename(fn)))