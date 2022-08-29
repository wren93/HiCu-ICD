Our data preprocessing follows [MultiResCNN](https://github.com/foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network) and [CAML](https://github.com/jamesmullenbach/caml-mimic) with slight modifications. To set up the dataset, follow the instructions below:

1. Download MIMIC-III dataset from [PhysioNet](https://physionet.org/content/mimiciii/1.4/).

2. Place the MIMIC-III files into `/data` as shown below:
```
data
|   D_ICD_DIAGNOSES.csv
|   D_ICD_PROCEDURES.csv
└───mimic3/
|   |   NOTEEVENTS.csv
|   |   DIAGNOSES_ICD.csv
|   |   PROCEDURES_ICD.csv
|   |   train_full_hadm_ids.csv
|   |   train_50_hadm_ids.csv
|   |   dev_full_hadm_ids.csv
|   |   dev_50_hadm_ids.csv
|   |   test_full_hadm_ids.csv
|   |   test_50_hadm_ids.csv
```
The `*_hadm_ids.csv` files can be found in the [CAML repository](https://github.com/jamesmullenbach/caml-mimic)

3. Run ```python preprocess_mimic3.py``` to preprocess the data.
