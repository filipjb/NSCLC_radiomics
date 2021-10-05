import numpy as np
import matplotlib.pyplot as plt
import pandas
import collections
import re
import radiomics
import pandas as pd
from patient_classes import Patient, StudyGroup
from radiomics import featureextractor, getTestCase, firstorder
import SimpleITK as sitk
from main import remove_disqualified_patients


def initiate_featureextractor(settings=dict):
    # Initializing the extractor
    extractor = featureextractor.RadiomicsFeatureExtractor(settings)
    # Disabling all features
    extractor.disableAllFeatures()
    return extractor


def calculate_firstorder_features(
        patient,
        filepath,
        featurelist=None,
        mute=True
):
    images = sitk.GetImageFromArray(patient.return_image_array(filepath))
    masks = sitk.GetImageFromArray(patient.return_GTV_segmentations(filepath))

    extractor = initiate_featureextractor()
    #extractor.enableFeaturesByName(firstorder=["Energy"])
    extractor.enableFeatureClassByName(featureClass="firstorder")

    print(f"\nCalculating first-order features for patient {patient}")
    # The feature vector returned by execute is a collections.OrderedDict
    feature_dict = extractor.execute(images, masks)
    if not mute:
        for featurename in feature_dict.keys():
            print(f"Computed {featurename}: {feature_dict[featurename]}")

    patient.firstorder_features = feature_dict
    return feature_dict


if __name__ == '__main__':
    csv_path = "pythondata/NSCLC Radiomics Lung1.clinical-version3-Oct 2019.csv"
    lung1_path = "C:/Users/filip/Desktop/image-data/manifest-Lung1/NSCLC-Radiomics"
    # 014, 021, 085 and 194 are excluded due errors in the files provided for these patients, 128 is excluded
    # due to no segmentatiion file being provded at all (post-operative case, acounted for in study)
    disq_patients = ["LUNG1-014", "LUNG1-021", "LUNG1-085", "LUNG1-194", "LUNG1-128", "LUNG1-095"]

    # Initiating our studygroup, adding all patients, and removing those that are excluded
    lung1 = StudyGroup()
    lung1.add_all_patients(csv_path)
    remove_disqualified_patients(lung1, disq_patients)

    dataframes = list()
    # Looping through every patient to calculate their features
    for patient in lung1:
        fo_features = calculate_firstorder_features(patient, lung1_path)
        # Creating a new dict inorder to sort out dict values that are not relevant and is not possible
        # to be added to a dataframe, i.e. the different elements containing metadata about the input image
        new_dict = dict()
        # If a key in the dict contains the word "diagnostics" it is not compatible with a dataframe, and is thus
        # not added to the dict that will be made a df
        for key in fo_features:
            if not re.search("diagnostics", key):
                new_dict.update({key: float(fo_features[key])})
        df = pd.DataFrame(new_dict, index=[lung1.index(patient)])
        dataframes.append(df)

    firstorder_featues = pd.concat(dataframes)

    

