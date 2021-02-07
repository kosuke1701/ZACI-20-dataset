# Danbooru 2020 Zero-shot Anime Character Identification Dataset (ZACI-20)

The goal of this dataset is creating human-level character identification models which do not require retraining on novel characters. The dataset is derived from [Danbooru2020 dataset](https://www.gwern.net/Danbooru2020).

## Features

* Large-scale
  - 1.45M images of 39K characters (train dataset).
* Designed for zero-shot setting.
  - Characters in the test dataset do not appear in the train dataset, allowing us to test model performance on novel characters.
* Human annotated test dataset.
  - Image pairs with errorneous face detection or duplicate images are manually removed.
  - We can compare model performance to human performance.

## Benchmarks

| model name | FPR (%) | FNR (%) | EER (%) | note |
|---|---|---|---|---|
| Human | 1.59 | 13.9 | N/A | by kosuke1701 |
| ResNet-152 | **2.40** | 13.9 | 8.89 | w/ RandAug, Contrastive loss. [0206_resnet152](https://github.com/kosuke1701/AnimeCV/releases/download/0111_best_randaug/0206_resnet152.zip) by kosuke1701 |
| SE-ResNet-152 | 2.43 | 13.9 | **8.15** | w/ RandAug, Contrastive loss. [0206_seresnet152](https://github.com/kosuke1701/AnimeCV/releases/download/0111_best_randaug/0206_seresnet152.zip) by kosuke1701 |
| ResNet-18 | 5.08 | 13.9 | 9.59 | w/ RandAug, Contrastive loss. [0206_resnet18](https://github.com/kosuke1701/AnimeCV/releases/download/0111_best_randaug/0206_resnet18.zip) by kosuke1701 |

* **Your participation is welcome!!** Please create an issue if you want to add your model to this list.
* Please do not use test dataset to tune hyperparameters!!
* You can use external resources. However, please do not use any data with character labels in test dataset to ensure fair comparison.
  - Note that `/` in original character labels of Danbooru 2020 is replaced by `__`.

## Getting Started
### Preprocess images

* Download SFW 512 px subset of [Danbooru 2020 dataset](https://www.gwern.net/Danbooru2020).
* Install dependencies.
  - `pip install tqdm pillow`
* Crop images using a preprocessing code.
  - ```shell
    # Danbooru 2020 SFW 512 px directroy
    export DANBOORU_DIR=/path/to/danbooru2020/512px
    # train
    python process_danbooru.py --danbooru-dir ${DANBOORU_DIR} --dataset-fn dataset/zaci20_train.json --save-dir zaci20_train
    # test
    python process_danbooru.py --danbooru-dir ${DANBOORU_DIR} --dataset-fn dataset/zaci20_test.json --save-dir zaci20_test
    ```
  - Images will be stored in different directories for each character.

### Evaluate your model

* Use evaluation code.
  - `python evaluate.py --test-pairs dataset/zaci20_test_pairs.csv --test-dataset-dir zaci20_test`
  - If you want to evaluate my benchmarks, download and unzip compressed model files. [AnimeCV]() should be installed to run my benchmarks.

## Notes

* Face annotations of [AnimeCV](https://github.com/kosuke1701/AnimeCV) is used to crop images.
  - See https://github.com/kosuke1701/AnimeCV/releases/tag/0.0 for more details.
* `/` in original character labels of Danbooru 2020 is replaced by `__`.
* I follow the methodology of [a previous work](https://github.com/grapeot/Danbooru2018AnimeCharacterRecognitionDataset) to construct this dataset.
* Benchmark models by me (kosuke1701) is trained with [this code](https://github.com/kosuke1701/optuna-metric-learning).
  - `python -u -m optuna_metric_learning.train --conf <CONFIG_FN> --model-def-fn examples/image_folder_example.py  --max-epoch 60 --patience 3 --n-fold 100`
  - Corresponding `<CONFIG_FN>`:
    - `tuned_configs/resnet18.json`
  - Results of conducted hyperparameter tuning on my private dataset is listed in [this Google spreadsheet](https://docs.google.com/spreadsheets/d/1kf4XnnEpWFugO--S1zD2lOv8jWyPAyYnL66POnNZKEM/edit?usp=sharing).

## Todo

- [ ] Create more difficult test dataset by adversarially sample negative image pairs.

## Citation

If you found this dataset or my benchmark models useful, please consider citing this repository and Danbooru 2020 dataset.

```
@misc{danbooru2020,
    author = {Anonymous and Danbooru community and Gwern Branwen},
    title = {Danbooru2020: A Large-Scale Crowdsourced and Tagged Anime Illustration Dataset},
    howpublished = {\url{https://www.gwern.net/Danbooru2020}},
    url = {https://www.gwern.net/Danbooru2020},
    type = {dataset},
    year = {2021},
    month = {January},
    timestamp = {2020-01-12},
    note = {Accessed: 2021-02-06} }
```

```
@misc{zaci20,
        author = {Kosuke Akimoto},
        title = {Danbooru 2020 Zero-shot Anime Character Identification Dataset (ZACI-20)},
        howpublished = {\url{https://github.com/kosuke1701/ZACI-20-dataset}},
        url = {https://github.com/kosuke1701/ZACI-20-dataset},
        type = {dataset,model},
        year = {2021},
        month = {February} }
```