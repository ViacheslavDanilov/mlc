import os
import pydicom
from tqdm import tqdm

# ----------------------------------------------------------------------------------------------------------------------
dir_to_check = 'data/coronal_sagittal/chest'
# ----------------------------------------------------------------------------------------------------------------------

def print_study_description(path):
    dcm = pydicom.dcmread(path)
    print('\nDICOM...............: {}'.format(path))
    if hasattr(dcm, 'StudyDescription'):
        print('Study description...: {}'.format(dcm.StudyDescription))
    else:
        print('Study description...: no description')

for root, dirs, files in tqdm(os.walk(dir_to_check), unit = 'dcm'):
    for file in files:
        if file.endswith(".dcm"):
            path = os.path.join(root, file)
            print_study_description(path)

dcm_path_1 = 'data/axial/abdomen/2-0.625s Axial  soft  DR 30-49870/000010.dcm'
print_study_description(dcm_path_1)

dcm_path_2 = 'data/axial/chest/2-ENTEROGRAPHY-78893/000306.dcm'
print_study_description(dcm_path_2)

dcm_path_3 = 'data/axial/pelvis/3-ABDPEL ST ss40-39579/000084.dcm'
print_study_description(dcm_path_3)

dcm_path_4 = 'data/coronal_sagittal/abdomen/604-MPR Thick Range2-48191/000118.dcm'
print_study_description(dcm_path_4)

dcm_path_5 = 'data/coronal_sagittal/abdomen_chest_pelvis/701-Sagittal-67437/000066.dcm'
print_study_description(dcm_path_5)

dcm_path_6 = 'ata/coronal_sagittal/chest/701-Display LUNG-25911/000021.dcm'
print_study_description(dcm_path_6)


