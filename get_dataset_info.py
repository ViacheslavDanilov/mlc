import os
import pydicom
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing.pool import Pool, ThreadPool
from sklearn.preprocessing import MultiLabelBinarizer

# -------------------------------------------------- Source parameters -------------------------------------------------
pool = ThreadPool(4)
data_dir = 'data'
save_dir = 'data'
file_ext = "dcm"
# ----------------------------------------------------------------------------------------------------------------------

# Check the number of DCM files
num_files = 0
for root, dirs, files in os.walk(data_dir):
    if not root == data_dir:
        num_files += len(files)
print(32*'-')
print('Total number of DCM files: {:d}'.format(num_files))
print(32*'-')

def process_dcm(dcm, filename):
    _path = filename
    view = filename.split(os.path.sep)[1]
    organ = filename.split(os.path.sep)[2].split('_')
    label = organ
    label.insert(0, view)
    classes = ['axial', 'coronal_sagittal', 'abdomen', 'chest', 'pelvis']
    mlb = MultiLabelBinarizer(classes=classes)
    mlb.fit(y=classes)
    out = mlb.transform([classes, label])[1]
    _axial = out[0]
    _coronal_sagittal = out[1]
    _abdomen = out[2]
    _chest = out[3]
    _pelvis = out[4]
    mc_label = np.array2string(out, precision=0, separator='', suppress_small=True)
    mc_label = mc_label[1:]
    _mc_label = mc_label[:-1]
    _id = dcm.PatientID
    age = dcm.PatientAge
    if not age:
        _age = 255
    else:
        age = age.replace('Y', '')
        _age = int(age)
    _sex = dcm.PatientSex
    _modality = dcm.Modality
    _height = int(dcm.Rows)
    _width = int(dcm.Columns)
    _mean = np.mean(dcm.pixel_array)
    _max = np.max(dcm.pixel_array)
    _min = np.min(dcm.pixel_array)
    return _path, _axial, _coronal_sagittal, _abdomen, _chest, _pelvis, _mc_label, \
           _id, _age, _sex, _modality, _height, _width,  _mean,  _max, _min

responses = []
for path, subdirs, files in tqdm(os.walk(data_dir), unit=' folder'):
    for name in files:
        if name.endswith(file_ext):
            filename = os.path.join(path, name)
            dcm = pydicom.dcmread(filename)
            # process_dcm(dcm=dcm, filename=filename)                               # Uncomment for debugging
            responses.append(pool.apply_async(process_dcm, (dcm, filename)))
pool.close()
pool.join()

# Create a pandas dataframe containing of the class infromation
column_order = ['path', 'axial', 'coronal_sagittal', 'abdomen', 'chest', 'pelvis', 'mc_label',
                'id', 'age', 'sex', 'modality', 'height', 'width', 'mean', 'max', 'min']
data_df = pd.DataFrame(columns=column_order)
paths = []
axials = []
coronal_sagittals = []
abdomens = []
chests = []
pelvises = []
mc_labels = []
ids = []
ages = []
sexs = []
modalities = []
heights = []
widths = []
means = []
maxs = []
mins = []

for response in tqdm(responses, unit=' file'):
    _path, _axial, _coronal_sagittal, _abdomen, _chest, _pelvis, _mc_label, \
    _id, _age, _sex, _modality, _height, _width, _mean, _max, _min = response.get()
    paths.append(_path)
    axials.append(_axial)
    coronal_sagittals.append(_coronal_sagittal)
    abdomens.append(_abdomen)
    chests.append(_chest)
    pelvises.append(_pelvis)
    mc_labels.append(_mc_label)
    ids.append(_id)
    ages.append(_age)
    sexs.append(_sex)
    modalities.append(_modality)
    heights.append(_height)
    widths.append(_width)
    means.append(_mean)
    maxs.append(_max)
    mins.append(_min)

data_df['path'] = pd.Series(paths)
data_df['axial'] = pd.Series(axials)
data_df['coronal_sagittal'] = pd.Series(coronal_sagittals)
data_df['abdomen'] = pd.Series(abdomens)
data_df['chest'] = pd.Series(chests)
data_df['pelvis'] = pd.Series(pelvises)
data_df['mc_label'] = pd.Series(mc_labels)

data_df['id'] = pd.Series(ids)
data_df['age'] = pd.Series(ages, dtype='int')
data_df['sex'] = pd.Series(sexs)
# sex_map = {'F': 0, 'M': 1}
# data_df['sex'] = data_df['sex'].replace(sex_map).astype('int')
data_df['modality'] = pd.Series(modalities)
data_df['height'] = pd.Series(heights)
data_df['width'] = pd.Series(widths)
data_df['mean'] = pd.Series(means)
data_df['max'] = pd.Series(maxs)
data_df['min'] = pd.Series(mins)

xlsx_name = os.path.join(save_dir, 'data.xlsx')
data_df.to_excel(xlsx_name, sheet_name='Data', index=True, startrow=0, startcol=0)
print('Сonversion completed!')