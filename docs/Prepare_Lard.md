# 1.  Prepare the original LARD dataset

First, download the LARD dataset from [LARD](https://github.com/deel-ai/LARD) repository. Then organize the original LARD dataset into a folder `LARD_dataset`, the structure of which should be like:

```shell
.
├── ImagerySources.txt
├── LARD_test
│   ├── LARD_test_real.zip
│   └── LARD_test_synth.zip
├── LARD_train
│   ├── LARD_train_BIRK_LFST.zip
│   ├── LARD_train_DAAG_DIAP.zip
│   ├── LARD_train_domain_adaptation.zip
│   ├── LARD_train_KMSY.zip
│   ├── LARD_train_LFMP_LFPO.zip
│   ├── LARD_train_LFQQ.zip
│   ├── LARD_train_LPPT_SRLI.zip
│   └── LARD_train_VABB.zip
├── LARD_train.csv
├── LICENSE
└── README.md

2 directories, 14 files
```

# 2. Unzip the original LARD

Run `src/data/1.unzip_data.py` to unzip all the above zip files to a new folder `LARD_dataset_decompressed`.

> You can manually unzip the files. The files after unzip should look like:

```shell
.
├── LARD_test
│   ├── LARD_test_real
│   │   ├── ImagerySources.txt
│   │   ├── LARD_test_real_edge_cases
│   │   ├── LARD_test_real_nominal_cases
│   │   └── LICENSE
│   └── LARD_test_synth
│       ├── ImagerySources.txt
│       ├── images
│       ├── infos.md
│       ├── LARD_test_synth.csv
│       ├── LARD_test_synth_scenarios.zip
│       └── LICENSE
└── LARD_train
    ├── LARD_train_BIRK_LFST
    │   ├── ImagerySources.txt
    │   ├── images
    │   ├── infos.md
    │   ├── LARD_train_BIRK_LFST.csv
    │   ├── LARD_train_BIRK_LFST_scenarios.zip
    │   └── LICENSE
    ├── LARD_train_DAAG_DIAP
    │   ├── ImagerySources.txt
    │   ├── images
    │   ├── infos.md
    │   ├── LARD_train_DAAG_DIAP.csv
    │   ├── LARD_train_DAAG_DIAP_scenarios.zip
    │   └── LICENSE
    ├── LARD_train_domain_adaptation
    │   ├── ImagerySources.txt
    │   ├── images
    │   ├── infos.md
    │   ├── LARD_train_domain_adaptation.csv
    │   ├── LARD_train_domain_adaptation_scenarios.zip
    │   └── LICENSE
    ├── LARD_train_KMSY
    │   ├── ImagerySources.txt
    │   ├── images
    │   ├── infos.md
    │   ├── LARD_train_KMSY.csv
    │   ├── LARD_train_KMSY_scenarios.zip
    │   └── LICENSE
    ├── LARD_train_LFMP_LFPO
    │   ├── ImagerySources.txt
    │   ├── images
    │   ├── infos.md
    │   ├── LARD_train_LFMP_LFPO.csv
    │   ├── LARD_train_LFMP_LFPO_scenarios.zip
    │   └── LICENSE
    ├── LARD_train_LFQQ
    │   ├── ImagerySources.txt
    │   ├── images
    │   ├── infos.md
    │   ├── LARD_train_LFQQ.csv
    │   ├── LARD_train_LFQQ_scenarios.zip
    │   └── LICENSE
    ├── LARD_train_LPPT_SRLI
    │   ├── ImagerySources.txt
    │   ├── images
    │   ├── infos.md
    │   ├── LARD_train_LPPT_SRLI.csv
    │   ├── LARD_train_LPPT_SRLI_scenarios.zip
    │   └── LICENSE
    └── LARD_train_VABB
        ├── ImagerySources.txt
        ├── images
        ├── infos.md
        ├── LARD_train_VABB.csv
        ├── LARD_train_VABB_scenarios.zip
        └── LICENSE

23 directories, 47 files
```

# 3. Convert to YOLO format

We use the code from LARD repository (with modifications) to convert the format.

Run  `src/data/2.data_2_yolo.py` to convert.

You should rename the corresponding file according the the following code:

```shell
cd ~/workspace/lard/lard-dataset/LARD_dataset_decompressed/LARD_test/LARD_test_real/
# for real nominal
mv LARD_test_real_nominal LARD_test_real_nominal_cases
cd LARD_test_real_nominal_cases
mv Test_Real_Nominal.csv LARD_test_real_nominal_cases.csv

# for real edge
cd ~/workspace/lard/lard-dataset/LARD_dataset_decompressed/LARD_test/LARD_test_real/LARD_test_real_edge_cases
mv Test_Real_Edge_Cases.csv LARD_test_real_edge_cases.csv
```

# 4. Remove the bad files

The following two samples have bad labels as reported by `YOLOv8` framework, and should be removed.

```python
bad_files = ['VABB_32_500_448', 'LWSK_34_500_132']
```

Run `src/data/3.remove_bad.py` to remove and backup the two samples.

# 5. Split the training and validation sets

We preserve 20% of the original (synthetic) training set as validation set.

Run `src/data/3.train_val_split.py to finish the split.`

# 6. Convert YOLO to coco

This is for better evaluation latter.

Run `src/data/4.yolo_2_coco.py` to convert.

# 7. Add soft link

You can run `src/data/5.add_link.py` to add a soft link between the project folder and data folder.

# 8. Modify absolute path

For each dataset `yaml` file in `cfg/ultralytics/datasets`, make sure to change the `path` variable to where the dataset is.

```yaml
path: /home/yeli/workspace/lard/lard-detection/datasets/lard/detection
train: train_all/images
val: test_real_edge/images
test: test_real_edge/images

nc: 1
names:
  0: runway
```

# 9. Final structure of datasets folder

The finall structure of the `datasets/lard` should look like:

```bash
.
├── annotations -> /fileonssd/runway-dataset/lard-dataset/annotations
│   ├── instances_test.json
│   ├── instances_test_real_edge.json
│   ├── instances_test_real.json
│   ├── instances_test_real_nominal.json
│   ├── instances_test_synth.json
│   ├── instances_train_all.json
│   ├── instances_train.json
│   └── instances_val.json
└── detection -> /fileonssd/runway-dataset/lard-dataset/YoloFormat/detection
    ├── bad_files
    │   ├── LWSK_34_500_132.jpeg
    │   ├── LWSK_34_500_132.txt
    │   ├── VABB_32_500_448.jpeg
    │   └── VABB_32_500_448.txt
    ├── test
    │   ├── images
    │   ├── labels
    │   └── labels.cache
    ├── test_real
    │   ├── images
    │   ├── labels
    │   └── labels.cache
    ├── test_real_edge
    │   ├── images
    │   ├── labels
    │   └── labels.cache
    ├── test_real_nominal
    │   ├── images
    │   ├── labels
    │   └── labels.cache
    ├── test_synth
    │   ├── images
    │   ├── labels
    │   └── labels.cache
    ├── train
    │   ├── images
    │   ├── labels
    │   └── labels.cache
    ├── train_all
    │   ├── images
    │   └── labels
    └── val
        ├── images
        ├── labels
        └── labels.cache

27 directories, 19 files
```
