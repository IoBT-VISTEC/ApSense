# ApSense: Data-driven Algorithm in PPG-based Sleep Apnea Sensing
This repository contains the official implementation of the paper: "ApSense: Data-driven Algorithm in PPG-based Sleep Apnea Sensing"

## Overview

ApSense is a novel deep learning approach for detecting Obstructive Sleep Apnea (OSA) events using fingertip Photoplethysmography (PPG) signals. The model demonstrates superior performance in OSA event recognition, particularly on high-variance datasets, while maintaining a relatively small model size.

## Data Preparation

### 1. Data EDF Extraction

#### MESA Dataset
1. Request access to the [MESA dataset](https://sleepdata.org/datasets/mesa)
2. Download the polysomnography (PSG) EDF files
3. Use the provided `1_EDF_File_Extract_MESA.ipynb` script to extract PPG signals:


#### HeartBEAT Dataset
1. Request access to the [HeartBEAT dataset](https://sleepdata.org/datasets/heartbeat)
2. Download the PSG EDF files
3. Use the provided `1_EDF_File_Extract_HeartBEAT.ipynb` script to extract PPG signals:

### 2. Feature Extraction & Annotation

#### PPG Feature Extraction
Run the feature extraction script to process the raw PPG signals:
`2_Data_prep_MESA.ipynb` and `2_Data_prep_HeartBEAT.ipynb`
This script extracts the following features:
- Pulse Wave Amplitude (PWA)
- PP Interval (PPI)
- Derivative of PWA (dPWA)
- Derivative of PPI (dPPI)
- Systolic Phase Duration (SPD)
- Diastolic Phase Duration (DPD)
- Pulse Area (PA)

#### Annotation Process
The annotation process:
1. Segments PPG signals into 60-second windows with 50% overlap
2. Marks OSA events based on specialist annotations
3. Aligns annotations with PPG windows

## Model Training and Evaluation

### Training

Train the ApSense model using:
```bash
python -u main.py \
	--dataset mesa \
	--model $YOUR_MODEL \
	--dataset_dir $PATH_TO_YOUR_PROCESSED_DATASET \
	--log_dir $LOG_DIR \
	--weight_dir $WEIGHT_DIR \
	--subsampling \
	--gpu GPU_NUMBER \
	> "stdout/mesa_${YOUR_MODEL}_aug.log" &
```

Configuration options in `models/dsepnet.py`:
- Number of RVarDSepBlocks (M)
- LSTM Variational Dropout settings
- Branch CNN configuration

### Evaluation

Evaluate the trained model:
```bash
python evaluate.py --dataset mesa \
                   --model $YOUR_MODEL \
                   --dataset_dir $PATH_TO_YOUR_PROCESSED_DATASET 
```

The evaluation script reports:
- Accuracy
- Macro F1 Score
- Sensitivity
- Specificity
- AUROC

## Citation
If you find this work useful, please cite our paper:
```bibtex
@article{choksatchawathi2024apsense,
  title={ApSense: Data-Driven Algorithm in PPG-Based Sleep Apnea Sensing},
  author={Choksatchawathi, Tanut and Sawadwuthikul, Guntitat and Thuwajit, Punnawish and Kaewlee, Thitikorn and Mateepithaktham, Thee and Saisaard, Siraphop and Sudhawiyangkul, Thapanun and Chaitusaney, Busarakum and Saengmolee, Wanumaidah and Wilaiprasitporn, Theerawit},
  journal={IEEE Internet of Things Journal},
  year={2024},
  publisher={IEEE}
}
```

## Acknowledgments

We thank the MESA and HeartBEAT studies for providing the datasets used in this research.
