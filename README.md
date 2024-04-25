# Weakly-Supervised Explainable Infection Severity Classification from Chest CT Scans

## Introduction
This is the official PyTorch implementation for our work titled "Weakly-Supervised Explainable Infection Severity Classification from Chest CT Scans". This implementation includes the following:
1. A multi-stage pipeline to extract features of low-level infection patterns as well as high-level infection coverage within the CT volume. 
2. A weakly-supervised approach to train the multi-stage pipeline to classify infection severities and to highlight infection regions for explainability. 
3. A novel proximity factor between local infection clusters to encourage more accurate and faster-converging infection localisation. 

We also include the code for training and testing other baseline 3D-CNN models on the same task. 

<img src="figures/weak_sup_inf_sev_method_diagram.svg"  width="600" height="500">

## Getting started
First start by cloning the repository using:

```bash
git clone git@github.com:BioMedIA-MBZUAI/InfectionSeverity-LungCT.git
```

### Dependencies
We recommend using Anacodna to create a virtual environment:

```bash
conda create inf_sev_env python=3.8
```

Install the requirements to your environment using the following command:

```bash 
pip install -r requirements
```

### Datasets
There are two main datasets that we use in this work, which you can download using the following:
1. [HUST Dataset](https://ngdc.cncb.ac.cn/ictcf/HUST-19.php): we exclude the suspected cases from this dataset, which leaves us with 1,063 CT scans.
2. [MosMed Dataset](https://mosmed.ai/en/): we use 1,110 CT scans from this dataset, along with 50 scans from mild cases which include segmentation masks that are used to evaluate the model's explainability.

### Project Strucutre

The project structure shown below contains three main modules:
1. Experiments folder which has the main training, validation, and testing code for the vanilla infection severity classification ([./experiments/infection_classification.py](./experiments/infection_classification.py)) and for our weakly supervised approach ([./experiments/region_proposal.py](./experiments/region_proposal.py)).
2. Modelling folder containing the different models implemented/adapted for this project.
3. Data folder containing the different dataset classes: MosMed, HUST, and their combindation in [./data/mos_hust.py](./data/mos_hust.py).
```
├── data
│   ├── dataset.py
│   ├── hust.py
│   ├── mos_hust.py
│   ├── mosmeddata.py
│   ├── transforms.py
│   └── utils
├── experiments
|   ├── batch.py
│   ├── experiment.py
│   ├── infection_classification.py
│   ├── region_proposal.py
│   └── utils
├── main.py
├── modelling
│   ├── ThreeDCNN.py
│   ├── densenet.py
│   ├── mlp.py
│   ├── region_prop.py
│   ├── region_prop_pipe.py
│   └── resnet.py
├── params
├── requirements.txt
└── utils
```

## Reproducing Results

### Training
You can test this first by running one of the sample parameter files provided:
```bash
python main.py -p ./params/experiments/vanilla/hust.json
```
To train any of the models for infection severity classification, you can carry out the following steps:
1. Create a parameters JSON file for the experiment (e.g. ). You can follow one of the examples parater files available in [./params/experiments](./params/experiments).
2. Run the experiment(s) using:
```bash
python main.py -p ./params/
```

It is important to note that the 3D DenseNet models use 3D average pooling layers, which do not have a deterministic backpropgation implementation. Therefore, the results we report here and in the paper are an aggregated result of multiple runs.

### Inference


### Explainiblity


# Results
The table below contains the results obtained along with the relevant links to reproduce the results. 
| Dataset | No. Classes | Method | Hyperparams | F1 |
| :--- |  :---: |  :--- | :--- | :---: |
| HUST | 3 | [AD3D-MIL](https://github.com/ibrahimalmakky/AD3DMIL-InfectionSeverity)<br>Slice-wise VGG-16<br>3D-DenseNet-121<br>Ours - DenseNet-121 Backbone | [trainval_mc.yaml](https://github.com/ibrahimalmakky/AD3DMIL-InfectionSeverity/blob/master/cfgs/trainval_mc.yaml)<br><br>[vanilla/hust.json](params/experiments/vanilla/hust.json)<br>[weak_sup/hust.json](params/experiments/weak_sup/hust.json) | 0.41<br>0.5748<br>0.5772 $\pm$ 0.0216<br>**0.6020** $\pm$ 0.0260 |
| MosMed | 3 | [AD3D-MIL](https://github.com/ibrahimalmakky/AD3DMIL-InfectionSeverity)<br>3D-DenseNet-121<br>Ours - DenseNet-121 Backbone  | [trainval_mc_mosmed.yaml](https://github.com/ibrahimalmakky/AD3DMIL-InfectionSeverity/blob/master/cfgs/trainval_mc_mosmed.yaml)<br>[vanilla/mosmed.json](params/experiments/vanilla/mosmed.json)<br>[weak_sup/mosmed.json](params/experiments/weak_sup/mosmed.json) | 0.6161<br>0.5928 $\pm$ 0.0101<br>**0.6228** $\pm$ 0.0291|
| MosMed+HUST | 3 | [AD3D-MIL](https://github.com/ibrahimalmakky/AD3DMIL-InfectionSeverity)<br>3D-DenseNet-121<br>Ours - DenseNet-121 Backbone | [trainval_mc_moshust.yaml](https://github.com/ibrahimalmakky/AD3DMIL-InfectionSeverity/blob/master/cfgs/trainval_mc_moshust.yaml)<br>[vanilla/moshust_3class.json](params/experiments/vanilla/moshust_3class.json)<br>[weak_sup/moshust_3class.json](params/experiments/weak_sup/moshust_3class.json) | 0.5988<br>**0.6648** $\pm$ 0.0058<br> 0.6273 $\pm$ 0.0264 |
| MosMed+HUST | 5 | [AD3D-MIL](https://github.com/ibrahimalmakky/AD3DMIL-InfectionSeverity)<br>3D-DenseNet-121<br>Ours - DenseNet-121 Backbone | [trainval_mc_moshust_5class.yaml](https://github.com/ibrahimalmakky/AD3DMIL-InfectionSeverity/blob/master/cfgs/trainval_mc_moshust_5class.yaml)<br>[vanilla/moshust_5class.json](params/experiments/vanilla/moshust_5class.json)<br>[weak_sup/moshust_5class.json](params/experiments/weak_sup/moshust_5class.json) | 0.4743<br> **0.5968** $\pm$ 0.0103 <br> 0.5081 $\pm$ 0.0220 |

# Citation
If you wish to use any part of this code, please cite our work using:


# Contact
If you have any questions about this code, please get in touch throug email @ ibrahim.almakky@mbzuai.ac.ae.
