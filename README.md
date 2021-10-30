# Monte Carlo Concrete DropPathfor Epistemic Uncertainty Estimationin Brain Tumor Segmentation

This repository is the official implementation of 
[Monte Carlo Concrete DropPath for Epistemic Uncertainty Estimation in Brain Tumor Segmentation](https://link.springer.com/chapter/10.1007/978-3-030-87735-4_7#Sec7). 

## Requirements

We used python 3.8.\
To install requirements run:

```setup
pip install -r requirements.txt
```
To run models unzip data and test data into `data` and `data_test` folders respectively or change paths to the
directories in .sh files.

## Training

To train **NASNet MC Concrete DropPath** run:
```bash
./train_nasnet_cdp.sh
```
To train another model change args `network` and train parameters respectively in .sh file.

Best NASNet MC CDP model has been trained for 600 epochs

## Evaluation

To test **NASNet MC Concrete DropPath** run:
```bash
./test_nasnet_cdp.sh
```
To test another model change args `network` `dropout_rate` and `model` respectively in .sh file
