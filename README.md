# cse6250_bitenet
Dependencies - found in requirements.txt

Data - Follow the steps from the MIMIC III site (https://physionet.org/content/mimiciii/1.4/) to get access to MIMIC III GCP. From there, download the following files -admissions.csv, diagnoses_icd.csv, procedures_icd.csv

Run Code 
- create environment using conda create --name <env> --file requirements.txt
- save csv files in data folder (for ex. mimic3_data)
- in main.py, update file names and dataset_path
- run python main.py in terminal
  
Functionality
- dataset.py - contains data_prep function for data processing and MedDataset class to convert to Tensor Dataset
- model.py - contains neural network class to build model
- main.py - reads in and processes the data, trains the model, and shows results
