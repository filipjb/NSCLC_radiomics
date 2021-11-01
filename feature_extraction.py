import numpy as np
import matplotlib.pyplot as plt
import pandas
import collections
import re
import radiomics
import pandas as pd
from patient_classes import Patient, StudyGroup
from radiomics import featureextractor, getTestCase, firstorder, shape
import SimpleITK as sitk
from main import remove_disqualified_patients


def initiate_featureextractor(settings=dict):
    # Initializing the extractor
    extractor = featureextractor.RadiomicsFeatureExtractor(settings)
    # Disabling all features
    extractor.disableAllFeatures()
    return extractor


def calculate_firstorder_features(patient_group, filepath, featurelist=None, mute=True):
    # A list for constructing the final total dataframe that will be returned to the user
    dataframes = list()

    # Looping through every patient in the group to calcualate each patients featurevalues
    for patient in patient_group:
        # Retrieving images and segmentations from the patient
        images = sitk.GetImageFromArray(patient.return_image_array(filepath))
        masks = sitk.GetImageFromArray(patient.return_GTV_segmentations(filepath))

        # Enabling features and extracting firstorder radiomic feature values
        firstorder_features = firstorder.RadiomicsFirstOrder(images, masks)
        firstorder_features.enableAllFeatures()
        # Standard deviation is not enabled by enableAllFeatures due to realation to other features
        firstorder_features.enableFeatureByName("StandardDeviation", True)
        firstorder_features.execute()

        print(f"\nCalculating first-order features for patient {patient}")
        if not mute:
            for featurename in firstorder_features.featureValues.keys():
                print(f"Computed {featurename}: {firstorder_features.featureValues[featurename]}")
        # Turning the dict into a dataframe
        df = pd.DataFrame(firstorder_features.featureValues, index=[patient_group.index(patient)])
        # And appending the dataframe to the list of all dataframes
        dataframes.append(df)
    # Concatenating the list of dataframes into a single dataframe containing all features of all patients
    features_df = pd.concat(dataframes)
    # Returning the final dataframe
    return features_df


def calculate_shape_features(patient_group, filepath, featurelist=None, mute=True):
    # a list for constructing the final total dataframe that will be retured to the use
    dataframes = list()

    # Looping through every patient in the group and calculating each patients feature values
    for patient in patient_group:
        # Retrieving images and segmentations from the patient
        images = sitk.GetImageFromArray(patient.return_image_array(filepath))
        masks = sitk.GetImageFromArray(patient.return_GTV_segmentations(filepath))

        # Enabling features and extracting the patient's radiomic shape features
        shape_features = shape.RadiomicsShape(images, masks)
        shape_features.enableAllFeatures()
        # Compactness is not enabled by enableAllFeatures due to relation to other features
        shape_features.enableFeatureByName("Compactness2", True)
        shape_features.execute()

        print(f"Calculating shape features for patient {patient}")
        if not mute:
            for featurename in shape_features.featureValues.keys():
                print(f"Computed {featurename}: {shape_features.featureValues[featurename]}")

        df = pd.DataFrame(shape_features.featureValues, index=[patient_group.index(patient)])
        dataframes.append(df)

    features_df = pd.concat(dataframes)

    return features_df


def calculate_texture_features(patient_group, filepath, featurelist=None, mute=True):
    pass


def calculate_wavelet_features(patient_group, filepath, featurelist=None, mute=True):
    pass


#%%
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

    sub = lung1[0:5]

    frame = calculate_firstorder_features(sub, lung1_path, mute=False)
    print(frame)
