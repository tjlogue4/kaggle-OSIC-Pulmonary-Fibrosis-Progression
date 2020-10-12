import pandas as pd
import pydicom
import numpy as np
import os
import cv2
from tqdm import tqdm
import multiprocessing as mp

cpu_count = mp.cpu_count() #gets the number of cores your machine has



NEW_SIZE = 512

DATA_DIR = 'F:/OISC/osic-pulmonary-fibrosis-progression/train/'
SAVE_DIR = 'C:/temp/processed_files/'
PATIENTS = os.listdir(DATA_DIR)

df_train = pd.read_csv('F:/OISC/osic-pulmonary-fibrosis-progression/train.csv')

p_amount = len(PATIENTS)

errors = pd.DataFrame(columns = ['Patient', 'Error']) 
errors.to_csv('error_log.csv', index = False)

#slope
for patient in PATIENTS:
    x = df_train[df_train["Patient"]==patient]["Weeks"] 
    y = df_train[df_train["Patient"]==patient]["FVC"]
    slope = np.polyfit(x, y, 1)[0]
    df_train.loc[df_train["Patient"]==patient, 'Slope'] = slope
del df_train['Weeks']
del df_train['FVC']
del df_train["Percent"]
df_train = df_train.drop_duplicates()
df_train['label'] = 0
df_train.loc[df_train['Slope'] < -3,'label'] = 1

#######################################################################################

#code from https://www.kaggle.com/allunia/pulmonary-dicom-preprocessing

def set_outside_scanner_to_air(raw_pixelarrays):
    # in OSIC we find outside-scanner-regions with raw-values of -2000. 
    # Let's threshold between air (0) and this default (-2000) using -1000
    raw_pixelarrays[raw_pixelarrays <= -1000] = 0
    return raw_pixelarrays

def transform_to_hu(slices):
    images = np.stack([file.pixel_array for file in slices])
    images = images.astype(np.int16)

    images = set_outside_scanner_to_air(images)
    
    # convert to HU
    for n in range(len(slices)):
        
        intercept = slices[n].RescaleIntercept
        slope = slices[n].RescaleSlope
        
        if slope != 1:
            images[n] = slope * images[n].astype(np.float64)
            images[n] = images[n].astype(np.int16)
            
        images[n] += np.int16(intercept)
    
    return np.array(images, dtype=np.int16)

#######################################################################################




def multi(p= patient, data_dir = DATA_DIR, patients = PATIENTS, new_size = NEW_SIZE):

    
    try:
        path = data_dir + p 
        slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)] 
        slices.sort(key = lambda x: float(x.ImagePositionPatient[2])) #step 1
        slices = transform_to_hu(slices)#step 3
        slices = [cv2.resize(np.array(each_slice), (new_size, new_size)) for each_slice in slices] #step 4
        label = int(df_train[df_train['Patient'] == p]['label'])#step 5 get the label value
        for num, each_slice in enumerate(slices): #notice i am now enumerating the slices 
            np.save(f'{SAVE_DIR}{p}{num}.npy', [each_slice, label]) #notice my save directory, much_data is also gone 
    except Exception as e:
        error_dict = {'Patient' : patient, 'Error': e}
        temp_error = pd.read_csv('error_log.csv') 
        temp_error = temp_error.append(error_dict, ignore_index = True)
        temp_error.to_csv('error_log.csv', index = False)
        
        





if __name__ == '__main__':

    
    pool = mp.Pool(cpu_count)
    for _ in tqdm(pool.imap_unordered(multi, [patient for patient in PATIENTS]), total = len(PATIENTS)):
        pass
    #results = pool.map(multi, [patient for patient in PATIENTS]) #here we call the funtion and the list we want to pass
    

    pool.close()
