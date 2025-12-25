import numpy as np
import json 
import matplotlib.pyplot as plt

# ----------------------Phase 1: sanity check ------------------------

evaluation_list = ['HNC_001+9Ag+MOS_25934', 'HNC_001+A4Ac+MOS_25934', '0617-259694+2Ac+MOS_33896', '0617-259694+imrt+MOS_33896' ] #  defined by organizers
PTVHighname_list = [ 'PTVHighOPT', 'PTVHighOPT', 'PTV', 'PTV']  # provided by organizers, lung site is PTV, HNC site is PTVHighOPT
reference_data_folder = '../../submission/data' # only accessable to organizers for Phase II and Phase III. 

prediction_folder = 'results' 


# ----------------------Phase 2: validation ------------------------
# import pandas as pd
# df = pd.read_csv('meta_files/meta_data.csv')
# df = df.loc[df['phase'] == 'valid']
# npz_paths = df['npz_path'].tolist()
# evaluation_list = [path.split('/')[-1].split('.')[0] for path in npz_paths]

# site_list = df['site'].tolist()
# PTVHighname_list = [ 'PTVHighOPT' if site == 1 else 'PTV' for site in site_list ]  # provided by organizers, lung site is PTV, HNC site is PTVHighOPT
# reference_data_folder = '../../data/GDP-HMM_Challenge/valid_withdose' # only accessable to organizers for Phase II and Phase III. 
# prediction_folder = '../pretrainmodel/GDP-HMM_Challenge/lightning/results' 

MAE_list = []

for i in range(len(evaluation_list)):

    plan_file_name = evaluation_list[i]
    PTVHighname = PTVHighname_list[i]

    patient_id = plan_file_name.split('+')[0]

    data_path = f'{reference_data_folder}/{plan_file_name}.npz'
    data_npz = np.load(data_path, allow_pickle=True)
    data_dict = dict(data_npz)['arr_0'].item()
    #data_dict = dict(data_npz)
    scale_dose_Dict = json.load(open('meta_files/PTV_DICT.json'))
    ref_dose = data_dict['dose'] * data_dict['dose_scale']
    ptv_highdose =  scale_dose_Dict[patient_id]['PTV_High']['PDose']
    norm_scale = ptv_highdose / (np.percentile(ref_dose[data_dict[PTVHighname].astype('bool')], 3) + 1e-5)
    ref_dose = ref_dose * norm_scale

    prediction = np.load(f'{prediction_folder}/{plan_file_name}_pred.npy')

    isodose_5Gy_mask = ((ref_dose > 5) | (prediction > 5)) & (data_dict['Body'] > 0) # the mask include the body AND the region where the dose/prediction is higher than 5Gy

    isodose_ref_5Gy_mask = (ref_dose > 5) & (data_dict['Body'] > 0) # the mask include the body AND the region where the ref is higher than 5Gy

    diff = ref_dose - prediction

    error = np.sum(np.abs(diff)[isodose_5Gy_mask > 0]) / np.sum(isodose_ref_5Gy_mask)
    print (error)
    MAE_list.append(error)

print ('the Metric 1 MAE displayed in leaderboard should be: ', np.mean(MAE_list))