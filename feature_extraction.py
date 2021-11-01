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


def calculate_firstorder_features(patient_group, filepath, featurelist=None, mute=True):
    extractor = initiate_featureextractor()
    # Enabling all firstorder features in the feature extractor
    extractor.enableFeatureClassByName(featureClass="firstorder")
    # A list for constructing the final total dataframe that will be returned to the user
    dataframes = list()

    # Looping through every patient in the group to calcualate each patients featurevalues
    for patient in patient_group:
        # Retrieving images and segmentations from the patient
        images = sitk.GetImageFromArray(patient.return_image_array(filepath))
        masks = sitk.GetImageFromArray(patient.return_GTV_segmentations(filepath))

        print(f"\nCalculating first-order features for patient {patient}")
        # The feature vector returned by execute is a collections.OrderedDict
        feature_dict = extractor.execute(images, masks)
        if not mute:
            for featurename in feature_dict.keys():
                print(f"Computed {featurename}: {feature_dict[featurename]}")

        # Creating a new dict in order to sort out dict values that are not relevant and is not possible
        # to be added to a dataframe (vectors), i.e. the different elements containing metadata about the input image
        new_dict = dict()
        # If a key in the dict contains the word "diagnostics" it is not compatible with a dataframe, and is thus
        # not added to the dict that will be made a df
        for key in feature_dict:
            if not re.search("diagnostics", key):
                new_dict.update({key: float(feature_dict[key])})
        # Making the new dict, only containing valid values, into a dataframe
        df = pd.DataFrame(new_dict, index=[lung1.index(patient)])
        # Appending this dataframe into our list of dataframes
        dataframes.append(df)
    # Concatenating the list of dataframes into a single dataframe containing all features of all patients
    features_df = pd.concat(dataframes)
    # Returning the final dataframe
    return features_df


def calculate_shape_features(patient_group, filepath, featurelist=None, mute=True):
    extractor = initiate_featureextractor()
    # Enabling all shape features in the extractor
    extractor.enableFeatureClassByName(featureClass="shape")
    extractor.enableFeaturesByName(shape=["compactness 1"])
    # a list for constructing the final total dataframe that will be retured to the use
    dataframes = list()

    # Looping through every patient in the group and calculating each patients feature values
    for patient in patient_group:
        # Retrieving images and segmentations from the patient
        images = sitk.GetImageFromArray(patient.return_image_array(filepath))
        masks = sitk.GetImageFromArray(patient.return_GTV_segmentations(filepath))

        print(f"Calculating shape features for patient {patient}")
        feature_dict = extractor.execute(images, masks)
        if not mute:
            for featurename in feature_dict.keys():
                print(f"Computed {featurename}: {feature_dict[featurename]}")

        new_dict = dict()
        for key in feature_dict:
            if not re.search("diagnostics", key):
                new_dict.update({key: float(feature_dict[key])})

        df = pd.DataFrame(new_dict, index=[lung1.index(patient)])
        dataframes.append(df)

    features_df = pd.concat(dataframes)

    return features_df




def calculate_texture_features(patient_group, filepath, featurelist=None, mute=True):
    pass


def calculate_wavelet_features(patient_group, filepath, featurelist=None, mute=True):
    pass


#%%
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

frame = calculate_shape_features(sub, lung1_path, mute=False)
print(frame)

#%%
