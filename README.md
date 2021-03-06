Kaggle OSIC Pulmonary Fibrosis Progression
===================================

I recently started my MSBA program at Cal Poly Pomona and had the opportunity to participate in a signature project. These projects are optional and the intent of the project is to gain new knowledge and skills. I have previously worked with DICOM images and have some experience with neural networks so I decided to make a few tutorials that our team could use to build a better understanding of preprocessing DICOM images and training neural networks.

Although we are not entering into the actual competition on Kaggle, this should be a great learning experience.

## Exploration:
* [Exploration.ipynb](https://github.com/tjlogue4/kaggle-OSIC-Pulmonary-Fibrosis-Progression/blob/master/Exploration/Exploration.ipynb) General tutorial and exploration of the DICOM files and dataset.
## Preprocessing:
* [Preprocess.ipynb](https://github.com/tjlogue4/kaggle-OSIC-Pulmonary-Fibrosis-Progression/blob/master/Preprocessing/Preprocess.ipynb) Preprocessing tutorial.
* [process.py](https://github.com/tjlogue4/kaggle-OSIC-Pulmonary-Fibrosis-Progression/blob/master/Preprocessing/process.py) Preprocessing file that uses multiprocessing to improve performance.
* [error_log.csv](https://github.com/tjlogue4/kaggle-OSIC-Pulmonary-Fibrosis-Progression/blob/master/Preprocessing/error_log.csv) Log of errors encountered while preprocessing.
## Training:
* [Training.ipynb](https://github.com/tjlogue4/kaggle-OSIC-Pulmonary-Fibrosis-Progression/blob/master/Training/Training.ipynb) Introduction into deep learning, Keras, and building/evaluating models.
* [train_all.py](https://github.com/tjlogue4/kaggle-OSIC-Pulmonary-Fibrosis-Progression/blob/master/Training/train_all.py) Used for training several models on a p3.2xlarge EC2 instance in AWS.
* [model_log_train1603151129.csv](https://github.com/tjlogue4/kaggle-OSIC-Pulmonary-Fibrosis-Progression/blob/master/Training/model_log_train1603151129.csv) Log of the evaluation of each model trained in AWS.