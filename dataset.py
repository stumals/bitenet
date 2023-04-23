#%%
import pandas as pd
import numpy as np
import os
import io
from utils import convert_to_high_level_icd9
import torch
import numpy as np
from data_prep import data_prep, get_last_code, dataset_encoding, target_to_onehot
from torch.utils.data import Dataset, TensorDataset


def last_visit(row):
    if row['visit_num'] == row['visit_count']:
        return True
    else:
        return False
    
def data_prep(dataset_path, admin_file, diag_file, procedures_file,
              min_visits=2, max_visits=10, code_count_threshold=5,
              onehot=True):
    
    icd9_code_categories = list(range(19))

    df_adm = pd.read_csv(os.getcwd() + datasets_path + admin_file, dtype=str)
    df_diags = pd.read_csv(os.getcwd() + datasets_path + diag_file, dtype=str)
    df_proced = pd.read_csv(os.getcwd() + datasets_path + procedures_file, dtype=str)
    df_diags = df_diags.dropna()
    df_proced = df_proced.dropna()
    df_adm.columns = df_adm.columns.str.lower()
    df_diags.columns = df_diags.columns.str.lower()
    df_proced.columns = df_proced.columns.str.lower()

    # remove icd9_codes less than threshold
    icd9_diag_count = df_diags['icd9_code'].value_counts().to_frame().reset_index() \
                            .rename(columns={'icd9_code':'count', 'index':'icd9_code'})
    df_diags2 = df_diags.merge(icd9_diag_count, how='left', on='icd9_code')
    df_diags2 = df_diags2[df_diags2['count']>=code_count_threshold]

    icd9_proced_count = df_proced['icd9_code'].value_counts().to_frame().reset_index() \
                            .rename(columns={'icd9_code':'count', 'index':'icd9_code'})
    df_proced2 = df_proced.merge(icd9_proced_count, how='left', on='icd9_code')
    df_proced2 = df_proced2[df_proced2['count']>=code_count_threshold]


    df_adm['admittime'] = pd.to_datetime(df_adm['admittime'])

    # convert icd9_codes to high level mapping
    df_diags2['icd9_diag'] = df_diags2['icd9_code'].apply(convert_to_high_level_icd9)
    df_proced2['icd9_proced'] = df_proced2['icd9_code'].apply(convert_to_high_level_icd9)
    if not onehot:
        diags = df_diags2['icd9_diag'].unique()
        proceds = df_proced2['icd9_proced'].unique()
        codes = np.sort(np.unique(np.concatenate((diags, proceds))))
        df_code_map = pd.DataFrame([codes, np.arange(start=1,stop=len(codes)+1)]).transpose() \
                        .rename(columns={0:'icd9_code', 1:'icd9_index'})
        icd9_map = df_code_map.set_index('icd9_code').to_dict()['icd9_index']
        df_diags2['icd9_diag'] = df_diags2['icd9_diag'].apply(lambda x: icd9_map[x])
        df_proced2['icd9_proced'] = df_proced2['icd9_proced'].apply(lambda x: icd9_map[x])
    

    df_diags3 = df_diags2.loc[:,['subject_id', 'hadm_id', 'icd9_diag']].dropna()
    df_adm2 = df_adm.loc[:,['hadm_id', 'admittime']]
    df = df_diags3.merge(df_adm2, how='left', on='hadm_id')

    df_proced3 = df_proced2.loc[:,['subject_id', 'hadm_id', 'icd9_proced']]
    df2 = df.merge(df_proced3, how='left', on=['subject_id','hadm_id']).fillna(-1)
    df2['icd9_code'] = df2[['icd9_diag', 'icd9_proced']].apply(set, axis=1).apply(lambda x: [i for i in x if i > -1])
    df3 = df2.loc[:,['subject_id', 'admittime', 'icd9_code']]
    if onehot:
        df4 = df3.groupby(['subject_id', 'admittime']).agg(sum).reset_index()
    else:
        df4 = df3.groupby(['subject_id', 'admittime']).agg(sum).reset_index()
        df4['icd9_code'] = df4['icd9_code'].apply(set)
        df4['icd9_code'] = df4['icd9_code'].apply(lambda x: [i for i in x])
        df4['icd9_code'] = df4['icd9_code'].apply(lambda x: x + [0]*(len(codes)-len(x)))
    
    df4 = df4.sort_values(by=['subject_id', 'admittime'])    
    df4['visit_num'] = df4.groupby('subject_id')['admittime'].rank()

    # keep visit data for each patient up to max_visits + 1
    # +1 to get target for patients w/ > max_visits
    df5 = df4[df4['visit_num'] <= (max_visits+1)]

    df_visit_counts = df5['subject_id'].value_counts().to_frame().reset_index() \
            .rename(columns={'subject_id':'visit_count', 'index':'subject_id'})
    df6 = df5.merge(df_visit_counts, how='left', on='subject_id')

    # remove patients with less than min_visits
    df6 = df6[df6['visit_count']>=min_visits]

    df7 = df6.copy()
    df7['last_visit'] = df7.apply(last_visit, axis=1)
    if onehot:
        df7['icd9_code'] =  df7['icd9_code'].apply(lambda x: np.unique(np.array(x)))
        df7['icd9_onehot'] = df7['icd9_code'].apply(lambda x: np.in1d(np.arange(19), x).astype(int))
        df_target = df7[df7['last_visit']==True].loc[:,['subject_id', 'icd9_onehot']]
        df_data = df7[df7['last_visit']==False].loc[:,['subject_id', 'icd9_onehot']].groupby('subject_id').agg(list)
    else:
        df_target = df7[df7['last_visit']==True].loc[:,['subject_id', 'icd9_code']]
        df_data = df7[df7['last_visit']==False].loc[:,['subject_id', 'icd9_code']].groupby('subject_id').agg(list)

    # make data into numpy array
    if onehot:
        data_list = df_data['icd9_onehot'].to_list()
        data_array = np.zeros((len(data_list),len(icd9_code_categories), max_visits))
    else:
        data_list = df_data['icd9_code'].to_list()
        data_array = np.zeros((len(data_list),len(codes), max_visits))
    

    for i, p_visits in enumerate(data_list):
        for j, visit in enumerate(p_visits):
            data_array[i,:,j] = visit

    # make targets into numpy array
    if onehot:
        target_list = df_target['icd9_onehot'].to_list()
        target_array = np.zeros((len(target_list), len(icd9_code_categories)))
    else:
        target_list = df_target['icd9_code'].to_list()
        target_array = np.zeros((len(target_list), len(codes)))

    for i, target in enumerate(target_list):
        target_array[i,:] = target

    return data_array, target_array


class MedDataset(Dataset):
     def __init__(self, data_array, target_array):
        intervals = np.moveaxis(data_array.copy(), 1, -1)[:, :, 0]
        self.targets = target_array
        target_torch = torch.from_numpy(target_array)
        dataset_torch = torch.from_numpy(data_array)
        intervals_torch = torch.from_numpy(intervals)
        self.data_tensor = TensorDataset(dataset_torch.transpose(1, -1), intervals_torch, target_torch)

     def __len__(self):
        return len(self.targets)

     def get_data(self):
        return self.data_tensor

if __name__ == '__main__':

    admin_file = 'admissions_full.csv'
    diag_file = 'diagnoses_icd_full.csv'
    procedures_file = 'procedures_full.csv'
    datasets_path = '.\\mimic3_full\\'

    min_visits = 2
    max_visits = 10
    code_count_threshold = 5

    data, target = data_prep(datasets_path, admin_file, diag_file, procedures_file,
              min_visits=2, max_visits=10, code_count_threshold=5,
              onehot=True)
    
    print(data.shape)
    print(target.shape)