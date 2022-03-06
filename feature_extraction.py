import logging
import os

import numpy as np
import pandas as pd
import radiomics

from patient_classes import Patient, StudyGroup
from radiomics import firstorder, shape, glcm, glrlm
import SimpleITK as sitk
import pywt

# Settings for the feature extractors, where at least the binWidth is confirmed to be 25 in the study
settings = {'binWidth': 25,
            'interpolator': sitk.sitkBSpline,
            'resampledPixelSpacing': None}


def calculate_firstorder_features(patient_group, filepath, filetype, struc="GTVp", mute=True):
    # Logging:
    current_dir = os.path.dirname(os.path.realpath(__file__))
    log_file = os.path.join(current_dir, rf"feature_files\{patient_group.groupID}_firstorder_log.txt")
    handler = logging.FileHandler(filename=log_file, mode="w")
    formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
    handler.setFormatter(formatter)
    radiomics.logger.addHandler(handler)
    radiomics.logger.setLevel(logging.DEBUG)

    # A list for constructing the final total dataframe that will be returned to the user
    dataframes = list()

    # Looping through every patient in the group to calcualate each patients featurevalues
    for patient in patient_group:
        # Retrieving images and segmentations from the patient
        if filetype == "TCIA":
            images = sitk.GetImageFromArray(patient.get_TCIA_images(filepath))
            masks = sitk.GetImageFromArray(patient.get_TCIA_GTV_segmentations(filepath))
        elif filetype == "HUH":
            images, masks = patient.get_haukeland_data(filepath, structure=struc)
            images, masks = sitk.GetImageFromArray(images), sitk.GetImageFromArray(masks)
        else:
            print("Error: Uncrecognized filetype")
            quit()

        print(f"\nCalculating first-order features for patient {patient}")

        # Enabling features and extracting firstorder radiomic feature values
        firstorder_features = firstorder.RadiomicsFirstOrder(images, masks)
        firstorder_features.enableAllFeatures()
        # Standard deviation is not enabled by enableAllFeatures due to realation to other features
        firstorder_features.enableFeatureByName("StandardDeviation", True)
        firstorder_features.execute()

        if not mute:
            for featurename in firstorder_features.featureValues.keys():
                print(f"Computed {featurename}: {firstorder_features.featureValues[featurename]}")
        # Turning the dict into a dataframe
        df = pd.DataFrame(firstorder_features.featureValues, index=[patient_group.index(patient)])
        df.insert(0, "PatientID", patient.patientID)
        # And appending the dataframe to the list of all dataframes
        dataframes.append(df)
    # Concatenating the list of dataframes into a single dataframe containing all features of all patients
    features_df = pd.concat(dataframes)
    # Returning the final dataframe
    pd.DataFrame.to_csv(
        features_df, os.path.join(current_dir, rf"feature_files\{patient_group.groupID}_firstorder.csv")
    )


def calculate_shape_features(patient_group, filepath, filetype, struc="GTVp", mute=True):

    current_dir = os.path.dirname(os.path.realpath(__file__))
    log_file = os.path.join(current_dir, rf"feature_files\{patient_group.groupID}_shape_log.txt")
    handler = logging.FileHandler(filename=log_file, mode="w")
    formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
    handler.setFormatter(formatter)
    radiomics.logger.addHandler(handler)
    radiomics.logger.setLevel(logging.DEBUG)

    # a list for constructing the final total dataframe that will be retured to the use
    dataframes = list()

    # Looping through every patient in the group and calculating each patients feature values
    for patient in patient_group:
        # Retrieving images and segmentations from the patient
        if filetype == "TCIA":
            images = sitk.GetImageFromArray(patient.get_TCIA_images(filepath))
            masks = sitk.GetImageFromArray(patient.get_TCIA_GTV_segmentations(filepath))
        elif filetype == "HUH":
            images, masks = patient.get_haukeland_data(filepath, structure=struc)
            images, masks = sitk.GetImageFromArray(images), sitk.GetImageFromArray(masks)
        else:
            print("Error: Uncrecognized filetype")
            quit()

        print(f"Calculating shape features for patient {patient}")

        # Enabling features and extracting the patient's radiomic shape features
        shape_features = shape.RadiomicsShape(images, masks, **settings)
        shape_features.enableAllFeatures()
        # Compactness is not enabled by enableAllFeatures due to relation to other features
        shape_features.enableFeatureByName("Compactness2", True)
        shape_features.execute()

        if not mute:
            for featurename in shape_features.featureValues.keys():
                print(f"Computed {featurename}: {shape_features.featureValues[featurename]}")

        df = pd.DataFrame(shape_features.featureValues, index=[patient_group.index(patient)])
        df.insert(0, "PatientID", patient.patientID)
        dataframes.append(df)

    features_df = pd.concat(dataframes)
    pd.DataFrame.to_csv(
        features_df, os.path.join(current_dir, rf"feature_files\{patient_group.groupID}_shape.csv")
    )


def calculate_GLCM_features(patient_group, filepath, filetype, struc, mute=True):

    current_dir = os.path.dirname(os.path.realpath(__file__))
    log_file = os.path.join(current_dir, rf"feature_files\{patient_group.groupID}_GLCM_log.txt")
    handler = logging.FileHandler(filename=log_file, mode="w")
    formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
    handler.setFormatter(formatter)
    radiomics.logger.addHandler(handler)
    radiomics.logger.setLevel(logging.DEBUG)

    dataframes = list()

    for patient in patient_group:
        if filetype == "TCIA":
            images = sitk.GetImageFromArray(patient.get_TCIA_images(filepath))
            masks = sitk.GetImageFromArray(patient.get_TCIA_GTV_segmentations(filepath))
        elif filetype == "HUH":
            images, masks = patient.get_haukeland_data(filepath, structure=struc)
            images, masks = sitk.GetImageFromArray(images), sitk.GetImageFromArray(masks)
        else:
            print("Error: Uncrecognized filetype")
            quit()

        print(f"Calculating GLCM features for patient {patient}")

        glcm_features = glcm.RadiomicsGLCM(images, masks, **settings)
        glcm_features.enableAllFeatures()
        glcm_features.execute()

        if not mute:
            for featurename in glcm_features.featureValues.keys():
                print(f"Computed {featurename}: {glcm_features.featureValues[featurename]}")

        df = pd.DataFrame(glcm_features.featureValues, index=[patient_group.index(patient)])
        df.insert(0, "PatientID", patient.patientID)
        dataframes.append(df)

    features_df = pd.concat(dataframes)
    pd.DataFrame.to_csv(
        features_df, os.path.join(current_dir, rf"feature_files\{patient_group.groupID}_GLCM.csv")
    )


def calculate_GLRLM_features(patient_group, filepath, filetype, struc, mute=True):

    current_dir = os.path.dirname(os.path.realpath(__file__))
    log_file = os.path.join(current_dir, rf"feature_files\{patient_group.groupID}_GLRLM_log.txt")
    handler = logging.FileHandler(filename=log_file, mode="w")
    formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
    handler.setFormatter(formatter)
    radiomics.logger.addHandler(handler)
    radiomics.logger.setLevel(logging.DEBUG)

    dataframes = list()

    for patient in patient_group:
        if filetype == "TCIA":
            images = sitk.GetImageFromArray(patient.get_TCIA_images(filepath))
            masks = sitk.GetImageFromArray(patient.get_TCIA_GTV_segmentations(filepath))
        elif filetype == "HUH":
            images, masks = patient.get_haukeland_data(filepath, structure=struc)
            images, masks = sitk.GetImageFromArray(images), sitk.GetImageFromArray(masks)
        else:
            print("Error: Uncrecognized filetype")
            quit()

        print(f"Calculating GLRLM features for patient {patient}")

        glrlm_features = glrlm.RadiomicsGLRLM(images, masks, **settings)
        glrlm_features.enableAllFeatures()
        glrlm_features.execute()

        if not mute:
            for featurename in glrlm_features.featureValues.keys():
                print(f"Computed {featurename}: {glrlm_features.featureValues[featurename]}")

        df = pd.DataFrame(glrlm_features.featureValues, index=[patient_group.index(patient)])
        df.insert(0, "PatientID", patient.patientID)
        dataframes.append(df)

    features_df = pd.concat(dataframes)
    pd.DataFrame.to_csv(
        features_df, os.path.join(current_dir, rf"feature_files\{patient_group.groupID}_GLRLM.csv")
    )


def calculate_HLHGLRLM_features(patient_group, filepath, filetype, struc, mute=True):

    current_dir = os.path.dirname(os.path.realpath(__file__))
    log_file = os.path.join(current_dir, rf"feature_files\{patient_group.groupID}_HLH_GLRLM_log.txt")
    handler = logging.FileHandler(filename=log_file, mode="w")
    formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
    handler.setFormatter(formatter)
    radiomics.logger.addHandler(handler)
    radiomics.logger.setLevel(logging.DEBUG)

    dataframes = list()

    for patient in patient_group:
        # CT images are not made into sitk images yet, as the wavelet transform uses numpy array
        if filetype == "TCIA":
            images = patient.get_TCIA_images(filepath)
            masks = patient.get_TCIA_GTV_segmentations(filepath)
        elif filetype == "HUH":
            images, masks = patient.get_haukeland_data(filepath, structure=struc)
        else:
            print("Error: Uncrecognized filetype")
            quit()
        # Transform must have even dimensional images to work, so if the number of slices
        # is not even, the image array is padded with an extra black slice
        slices, rows, cols = np.shape(images)
        if slices % 2 != 0:
            images = np.append(images, [np.zeros([rows, cols])], axis=0)
            masks = np.append(masks, [np.zeros([rows, cols])], axis=0)

        print(f"Calculating HLH texture features for patient {patient}")

        # Taking the stationary (undecemated) wavelet transform of the ct images,
        # which returns a list of 8 dicts, one dict for each decomposition
        decomp = pywt.swtn(images, "coif1", level=1)[0]
        # We are interested in the HLH (i.e. dad) decomposition of the image, as it is the one
        # used for the radiomic signature in the study
        HLH = decomp["dad"]

        wavelet_images = sitk.GetImageFromArray(HLH)
        sitkmasks = sitk.GetImageFromArray(masks)

        # The size of each decomposition is the same as the original, so we can use the same
        # masks for feature calculation
        glrlm_wavelet = glrlm.RadiomicsGLRLM(wavelet_images, sitkmasks, **settings)
        glrlm_wavelet.enableAllFeatures()
        glrlm_wavelet.execute()

        # Looping through the dict and changing featurenames to differentiate from regular texture featurenames
        new_dict = dict()
        for featurekey, featureval in glrlm_wavelet.featureValues.items():
            new_dict.update({"HLH " + featurekey: featureval})
            if not mute:
                print(f"Computed HLH {featurekey}: {featureval}")

        # Using the new dict as the basis for forming the complete dataframe
        df = pd.DataFrame(new_dict, index=[patient_group.index(patient)])
        df.insert(0, "PatientID", patient.patientID)
        dataframes.append(df)

    features_df = pd.concat(dataframes)
    pd.DataFrame.to_csv(
            features_df, os.path.join(current_dir, rf"feature_files\{patient_group.groupID}_HLH_GLRLM.csv")
        )

