
Abstract:

Chest-X ray (CXR) radiography can be used as a first-line triage process for non-COVID-19 patients who have similar symptoms. However,  
the similarity between features of CXR images of COVID-19 and pneumonia caused by other infections make the differential diagnosis by 
radiologists challenging. We hypothesized that machine learning-based classifiers can reliably distinguish the CXR images of COVID-19 
patients from other forms of pneumonia. We used feature extraction and dimensionality reduction methods to generate an efficient machine
learning classifier that can distinguish COVID-19 cases from non-COVID-19 cases with high accuracy and sensitivity. We propose that our
COVID-Classifier classifier can be used in conjunction with other tests for optimal allocation of hospital resources by rapid triage of
non-COVID-19 cases.  

Dataset:
COVID-> 140 X-ray images
Normal-> 140 X-ray images
Pneumonia-> 140 X-ray images

How to use:
1-Run "preprocess_images.py" to preprocess images done by resizing, normalization, adaptive histogram equalization

2-Run "extract_features.py" to create three feature pools for covid or normal or pneumonia datasets

3-Run "evaluate_features.py" to evaluate extracted features

4-Run "train_model.py" to train and then evaluate model  

Test results:

	        Precision	 Sensitivity	 F-score	 Support
COVOD-19	   96%	           100%	           0.98	           25
Normal	           88%	           100%	           0.94	           31
Pneumonia	   100%	           82%	           0.91	           28


