Kaggle OSIC Pulmonary Fibrosis Progression
===================================

I recently started my MSBA program at Cal Poly Pomona and had the opportunity to participate in a signature project. These projects are optional and the intent of the project is to gain new knowledge and skills. I have previously worked with DICOM images and have some experience with neural networks so I decided to make a few tutorials that our team could use to build a better understanding of preprocessing DICOM images and training neural networks.

Although we are not entering into the actual competition on Kaggle, this should be a great learning experience.

### Notebook and Python files:
* [Exploration.ipynb](https://github.com/tjlogue4/kaggle-OSIC-Pulmonary-Fibrosis-Progression/blob/master/Exploration.ipynb) General tutorial and exploration of the DICOM files and dataset.
* [Preprocess.ipynb](https://github.com/tjlogue4/kaggle-OSIC-Pulmonary-Fibrosis-Progression/blob/master/Preprocess.ipynb) Preprocessing tutorial.
* [process.py](https://github.com/tjlogue4/kaggle-OSIC-Pulmonary-Fibrosis-Progression/blob/master/process.py) Preprocessing file that uses multiprocessing to improve performance.
* [Training.ipynb](https://github.com/tjlogue4/kaggle-OSIC-Pulmonary-Fibrosis-Progression/blob/master/Training.ipynb) Introduction into deep learning, Keras, and building/ evaluating models.
* [train_all.py](https://github.com/tjlogue4/kaggle-OSIC-Pulmonary-Fibrosis-Progression/blob/master/train_all.py) Used for training several models on a p3.2xlarge EC2 instance in AWS (see model [results](https://github.com/tjlogue4/kaggle-OSIC-Pulmonary-Fibrosis-Progression/blob/master/model_log_train1603151129.csv))

### CSV files:
* [error_log.csv](https://github.com/tjlogue4/kaggle-OSIC-Pulmonary-Fibrosis-Progression/blob/master/error_log.csv) Log of errors encountered while preprocessing.
* [model_log_train1603151129.csv](https://github.com/tjlogue4/kaggle-OSIC-Pulmonary-Fibrosis-Progression/blob/master/model_log_train1603151129.csv) Log of the evaluation of each model trained in AWS.