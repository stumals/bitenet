#### Data Prep ####a
import pandas as pd
import numpy as np
import os
import io

# admin_file = io.BytesIO(uploaded['ADMISSIONS.csv'])
# diag_file = io.BytesIO(uploaded['DIAGNOSES_ICD.csv'])
# procedure_file = io.BytesIO(uploaded['PROCEDURES_ICD.csv'])
# prescript_file = io.BytesIO(uploaded['PRESCRIPTIONS.csv'])
# drug_file = io.BytesIO(uploaded['DRGCODES.csv'])

# datasets_path = '.\\mimic3_demo_data\\'


def data_prep(datasets_path, admin_file, diag_file, procedure_file, prescript_file, drug_file):

    df_adm = pd.read_csv(os.getcwd() + datasets_path + admin_file, dtype=str)
    df_diags = pd.read_csv(os.getcwd() + datasets_path + diag_file, dtype=str)
    df_procedures = pd.read_csv(os.getcwd() + datasets_path + procedure_file, dtype=str)
    df_prescripts = pd.read_csv(os.getcwd() + datasets_path + prescript_file, dtype=str)
    df_drugs = pd.read_csv(os.getcwd() + datasets_path + drug_file, dtype=str)

    # df_adm = pd.read_csv(admin_file, dtype=str)
    # df_diags = pd.read_csv(diag_file, dtype=str)
    # df_procedures = pd.read_csv(procedure_file, dtype=str)
    # df_prescripts = pd.read_csv(prescript_file, dtype=str)
    # df_drugs = pd.read_csv(drug_file, dtype=str)

    df_adm['admittime'] = pd.to_datetime(df_adm['admittime'])
    df_adm['dischtime'] = pd.to_datetime(df_adm['dischtime'])
    df_adm['num_days'] = (df_adm['dischtime'] - df_adm['admittime']).dt.days + 1
    df_adm = df_adm.rename(columns={'hospital_expire_flag':'death_flag'})

    #df_diags = df_diags.rename(columns={'icd9_code':'icd9_diag'})
    #df_diags['icd9_parent'] = df_diags['icd9_code'].apply(lambda x: x[:4] if x[0] == 'E' else x[:3])
    df_diags['icd9_category'] = df_diags['icd9_code'].apply(convert_to_high_level_icd9)
    df_diags['icd9_diag'] = df_diags[['seq_num', 'icd9_category']].apply(tuple, axis=1)
    df_procedures = df_procedures.rename(columns={'icd9_code':'icd9_proced'})

    icd9_diag_categories = list(range(19))

    df_diags2 = df_diags.loc[:,['subject_id', 'hadm_id', 'icd9_diag']].dropna()
    df_adm2 = df_adm.loc[:,['hadm_id', 'admittime', 'num_days', 'death_flag']]
    df = df_diags2.merge(df_adm2, how='left', on='hadm_id')

    df_procedures2 = df_procedures.loc[:,['hadm_id', 'icd9_proced']].dropna()
    df2 = df.merge(df_procedures2, how='left', on='hadm_id')

    df_prescripts2 = df_prescripts.loc[:,['hadm_id', 'ndc']].dropna()
    df_prescripts2 = df_prescripts2[df_prescripts2['ndc']!='0']
    df3 = df2.merge(df_prescripts2, how='left', on='hadm_id')

    df_drugs2 = df_drugs.loc[:,['hadm_id', 'drg_code']].dropna()
    df4 = df3.merge(df_drugs2, how='left', on='hadm_id')

    df5 = df4.groupby(['subject_id', 'hadm_id']).agg(set)

    def sort_list(row):
        row2 = sorted(row, key=lambda x: int(x[0]))
        return [x[1] for x in row2]

    df5['icd9_diag'] = df5['icd9_diag'].apply(list)
    df5['icd9_diag'] = df5['icd9_diag'].apply(sort_list)
    df5['icd9_diag'] = df5['icd9_diag'].apply(lambda x: list(dict.fromkeys(x)))
    df5['icd9_proced'] = df5['icd9_proced'].apply(list)
    df5['ndc'] = df5['ndc'].apply(list)
    df5['drg_code'] = df5['drg_code'].apply(list)
    #df5['admittime'] = df5['admittime'].apply(lambda x: list(x)[0].strftime("%Y-%m-%d"))
    df5['admittime'] = df5['admittime'].apply(lambda x: list(x)[0])
    df5['num_days'] = df5['num_days'].apply(lambda x: list(x)[0]).astype(str)

    df6 = df5.reset_index()
    df6['death_flag'] = df6['death_flag'].apply(lambda x: list(x)[0])

    data = []
    patient_ids = df6['subject_id'].unique()
    for p_id in patient_ids:
        patient = {}
        df_p_id = df6[df6['subject_id']==p_id].sort_values('admittime')
        adm_ids = df_p_id.loc[:,'hadm_id']
        visits = []
        for adm_id in adm_ids:
            df_visit = df6[(df6['subject_id']==p_id) & (df6['hadm_id']==adm_id)]
            visit = {}
            visit['admittime'] = df_visit['admittime'].values[0]
            visit['num_days'] = df_visit['num_days'].values[0]
            visit['death_flag'] = df_visit['death_flag'].values[0]
            visit['icd9_diags'] = df_visit['icd9_diag'].values[0]
            visit['icd9_procedures'] = df_visit['icd9_proced'].values[0]
            visit['prescriptions'] = df_visit['ndc'].values[0]
            visit['drugs'] = df_visit['drg_code'].values[0]
            visits.append(visit)
        patient[p_id] = visits
        data.append(patient)

    return (data, icd9_diag_categories)

def get_last_code(data):
    last_codes = []
    for patient in data:
        for k in patient.keys(): 
            last = patient[k][-1]['icd9_diags'].pop()
            last_codes.append((last))
    return last_codes

def dataset_encoding(data, icd9_categories, max_visits):
    max_visits = 5
    dataset = np.zeros((len(data), len(icd9_categories), max_visits))
    for i, patient in enumerate(data):
        for p_id in patient.keys():
            for j, visit in enumerate(patient[p_id]):
                if j > max_visits-1:
                    continue
                else:
                    check_array = np.arange(len(icd9_categories))
                    visit_array = np.array(visit['icd9_diags'])
                    dataset[i,:,j] = np.in1d(check_array, visit_array).astype(int)
    return dataset

def target_to_onehot(target, dataset):
  target_onehot = np.zeros((dataset.shape[0], dataset.shape[1]))
  for i, t in enumerate(target):
      check_array = np.arange(dataset.shape[1])
      target_array = np.array([t])
      target_onehot[i,:] = np.in1d(check_array, target_array).astype(int)
  return target_onehot

def convert_to_high_level_icd9(icd9_code):
    k = icd9_code[:3]
    if '001' <= k <= '139':
        return 0
    elif '140' <= k <= '239':
        return 1
    elif '240' <= k <= '279':
        return 2
    elif '280' <= k <= '289':
        return 3
    elif '290' <= k <= '319':
        return 4
    elif '320' <= k <= '389':
        return 5
    elif '390' <= k <= '459':
        return 6
    elif '460' <= k <= '519':
        return 7
    elif '520' <= k <= '579':
        return 8
    elif '580' <= k <= '629':
        return 9
    elif '630' <= k <= '679':
        return 10
    elif '680' <= k <= '709':
        return 11
    elif '710' <= k <= '739':
        return 12
    elif '740' <= k <= '759':
        return 13
    elif '760' <= k <= '779':
        return 14
    elif '780' <= k <= '799':
        return 15
    elif '800' <= k <= '999':
        return 16
    elif 'E00' <= k <= 'E99':
        return 17
    elif 'V01' <= k <= 'V90':
        return 18