The VocalBM_Androids repository encompasses the work of the Supervised Project Group 6: Fundamental Frequency as a Vocal Biomarker of Depressive 
Traits: a Three-Part V3-based Validation Framework for F0 Robustness."  This repository stores the files containing our processed data and programming pipelines. 
This project was completed by Jillian Brown, Maé Dugoua-Jacques, Monica Pasqualini, and Maria Popa.  We utilized the Androids Corpus (link: https://github.com/androidscorpus/data).

# Files Framework:
Our files have been organized into 4 directories:

## 1. CleanCorpus
   This is the directory focused on our denoising pipeline.  ProcessingPipe contains the code pertaining to our denoising pipeline, while the other two folders inside this directory store our denoised ("ok") files by task.
## 2. classification
   classification contains our logistic regression pipeline for clinical validation.
## 3. feature_extraction
   Contains the python pipelines and results of F0 extraction.
## 4. stats
   Contains and organizes the available metadata of the Androids Corpus.
