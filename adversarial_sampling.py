from argparse import ArgumentParser
from glob import glob
import os

import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

def load_image(filename):
    try:
        with open(filename, "rb") as f:
            image = Image.open(f)
            return image.convert("RGB")
    except UserWarning as e:
        print(filename)
        input("Something wrong happens while loading image: {} {}".format(filename, str(e)))

# Example Model definition
class Model(object):
    def __init__(self, dirname):
        import animecv

        self.encoder = animecv.general.create_OML_ImageFolder_Encoder(dirname)
        self.encoder.to("cuda")
    
    # img: PIL image
    def encode(self, img):
        vecs = self.encoder.encode([img]).detach().cpu().numpy()
        return vecs[0]

if __name__=="__main__":
    parser = ArgumentParser()

    parser.add_argument("--test-pairs", help="CSV file which lists test image pairs.")
    parser.add_argument("--test-dataset-dir", help="Directory of test images.")
    parser.add_argument("--ignore-list", default=None, help="List of images which should be ignored during pair sampling.")

    parser.add_argument("--out-fn", default="adversarial.csv")

    parser.add_argument("--n-negative", type=int, default=3000)

    args = parser.parse_args()

    if not os.path.exists(args.out_fn):
        if args.ignore_list is not None:
            df = pd.read_csv(args.ignore_list, header=None)
            ignore_list = set(df.values.flatten().tolist())
        else:
            ignore_list = set()

        # Generate adversarial negative pairs.
        model = Model("0206_resnet152")

        images = glob(os.path.join(args.test_dataset_dir, "**"), recursive=True)
        images = [fn for fn in images if os.path.isfile(fn)]
        labels = [fn.split(os.path.sep)[-2] for fn in images]

        vecs = []
        for fn in tqdm(images):
            img = load_image(fn)
            vecs.append(model.encode(img).reshape((1,-1)))
        vecs = np.concatenate(vecs, axis=0)

        scores = np.sum(vecs[:,np.newaxis,:] * vecs[np.newaxis,:,:], axis=2)

        negative_pairs = []
        n_img = scores.shape[0]
        sorted_idx = np.argsort(-scores, axis=None).tolist()
        strip_len = len(args.test_dataset_dir + os.path.sep)
        while len(negative_pairs) < args.n_negative:
            idx = sorted_idx.pop(0)
            i,j = idx // n_img, idx % n_img
            if i<=j:
                continue
            if labels[i] == labels[j]:
                continue
            if os.path.basename(images[i]) in ignore_list:
                continue
            if os.path.basename(images[j]) in ignore_list:
                continue
            negative_pairs.append((images[i][strip_len:], images[j][strip_len:], 0, -1, 0))

        # Reuse positive pairs.
        positive_pairs = []
        df = pd.read_csv(args.test_pairs)
        for pathA, pathB in df[df["label"]==1][["pathA", "pathB"]].values:
            #print(pathA, pathB)
            positive_pairs.append((pathA, pathB, 1, -1, 0))
        
        pairs = shuffle(positive_pairs + negative_pairs)

        df = pd.DataFrame(pairs, columns=["pathA", "pathB", "label", "human_prediction", "invalid"])
        df.to_csv(args.out_fn, index=False)
    else:
        print("Reload")
        df = pd.read_csv(args.out_fn)
    
    for i_row in tqdm(list(range(df.values.shape[0]))):
        pathA, pathB, label, pred, invalid = df.loc[i_row].values
        #print(pathA, pathB)
        if pred >= 0:
            continue
        else:
            im1 = np.array(Image.open(os.path.join(args.test_dataset_dir, pathA)))
            im2 = np.array(Image.open(os.path.join(args.test_dataset_dir, pathB)))
            ax = plt.subplot(1,2,1)
            ax.imshow(im1)
            ax = plt.subplot(1,2,2)
            ax.imshow(im2)
            plt.draw()
            plt.pause(0.001)
            cmd = input("correct?[y/n]: ")
            if cmd=="y":
                pred = 1
            elif cmd=="n":
                pred = 0
            else:
                pred = 0
                df.loc[i_row, "invalid"] = 1
            df.loc[i_row, "human_prediction"] = pred
            df.to_csv(args.out_fn, index=False)
            plt.close()