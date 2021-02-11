from argparse import ArgumentParser
import os

import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from PIL import Image

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
    
    # img1, img2: PIL image
    def score(self, img1, img2):
        vecs = self.encoder.encode([img1, img2]).detach().cpu().numpy()
        score = np.dot(vecs[0], vecs[1])
        return score

if __name__=="__main__":
    parser = ArgumentParser()

    parser.add_argument("--test-pairs", help="CSV file which lists test image pairs.")
    parser.add_argument("--test-dataset-dir", help="Directory of test images.")

    parser.add_argument("--target-fnr", type=float, default=0.139, help="Reference FNR used to compute FPR.")

    args = parser.parse_args()

    model = Model("0206_seresnet152")

    df = pd.read_csv(args.test_pairs)
    true_labels = df["label"].values
    ROOT_DIR = args.test_dataset_dir
    scores = []
    for pathA, pathB, label in tqdm(df[["pathA", "pathB", "label"]].values):
        img1 = load_image(os.path.join(args.test_dataset_dir, pathA))
        img2 = load_image(os.path.join(args.test_dataset_dir, pathB))
        
        score = model.score(img1, img2)
        scores.append(score)
    
    fpr, tpr, threshold = roc_curve(true_labels, scores)
    eer = 1. - brentq(lambda x: 1. - x - interp1d(tpr, fpr)(x), 0., 1.)
    fnr = 1. - tpr
    print("False Positive Rate: ", interp1d(fnr, fpr)(args.target_fnr))
    print("Threshold: ", interp1d(fnr, threshold)(args.target_fnr))
    print("Equal Error Rate: ", eer)