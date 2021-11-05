import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from patient_classes import Patient, StudyGroup
from radiomics import firstorder, shape, glcm, glrlm
import SimpleITK as sitk
from main import remove_disqualified_patients
import pywt

# Settings for the feature extractors, where at least the binWidth is confirmed to be 25 in the study
settings = {'binWidth': 25,
            'interpolator': sitk.sitkBSpline,
            'resampledPixelSpacing': None}


def calculate_firstorder_features(patient_group, filepath, mute=True):
    # A list for constructing the final total dataframe that will be returned to the user
    dataframes = list()

    # Looping through every patient in the group to calcualate each patients featurevalues
    for patient in patient_group:
        # Retrieving images and segmentations from the patient
        images = sitk.GetImageFromArray(patient.return_image_array(filepath))
        masks = sitk.GetImageFromArray(patient.return_GTV_segmentations(filepath))

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
        # And appending the dataframe to the list of all dataframes
        dataframes.append(df)
    # Concatenating the list of dataframes into a single dataframe containing all features of all patients
    features_df = pd.concat(dataframes)
    # Returning the final dataframe
    return features_df


def calculate_shape_features(patient_group, filepath, mute=True):
    # a list for constructing the final total dataframe that will be retured to the use
    dataframes = list()

    # Looping through every patient in the group and calculating each patients feature values
    for patient in patient_group:
        # Retrieving images and segmentations from the patient
        images = sitk.GetImageFromArray(patient.return_image_array(filepath))
        masks = sitk.GetImageFromArray(patient.return_GTV_segmentations(filepath))

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
        dataframes.append(df)

    features_df = pd.concat(dataframes)

    return features_df


def calculate_GLCM_features(patient_group, filepath, mute=True):
    dataframes = list()

    for patient in patient_group:
        images = sitk.GetImageFromArray(patient.return_image_array(filepath))
        masks = sitk.GetImageFromArray(patient.return_GTV_segmentations(filepath))

        print(f"Calculating GLCM features for patient {patient}")

        glcm_features = glcm.RadiomicsGLCM(images, masks, **settings)
        glcm_features.enableAllFeatures()
        glcm_features.execute()

        if not mute:
            for featurename in glcm_features.featureValues.keys():
                print(f"Computed {featurename}: {glcm_features.featureValues[featurename]}")

        df = pd.DataFrame(glcm_features.featureValues, index=[patient_group.index(patient)])
        dataframes.append(df)

    features_df = pd.concat(dataframes)

    return features_df


def calculate_GLRLM_features(patient_group, filepath, mute=True):
    dataframes = list()

    for patient in patient_group:
        images = sitk.GetImageFromArray(patient.return_image_array(filepath))
        masks = sitk.GetImageFromArray(patient.return_GTV_segmentations(filepath))

        print(f"Calculating GLRLM features for patient {patient}")

        glrlm_features = glrlm.RadiomicsGLRLM(images, masks, **settings)
        glrlm_features.enableAllFeatures()
        glrlm_features.execute()

        if not mute:
            for featurename in glrlm_features.featureValues.keys():
                print(f"Computed {featurename}: {glrlm_features.featureValues[featurename]}")

        df = pd.DataFrame(glrlm_features.featureValues, index=[patient_group.index(patient)])
        dataframes.append(df)

    features_df = pd.concat(dataframes)

    return features_df


def calculate_waveletGLRLM_features(patient_group, filepath, mute=True):
    dataframes = list()

    for patient in patient_group:
        # CT images are not made into sitk images yet, as the wavelet transform uses numpy array
        images = patient.return_image_array(filepath)
        masks = patient.return_GTV_segmentations(filepath)

        # Transform must have even dimensional images to work, so if the number of slices
        # is not even, the image array is padded with an extra black slice
        slices, rows, cols = np.shape(images)
        if slices % 2 != 0:
            images = np.append(images, [np.zeros([rows, cols])], axis=0)
            # Masks must of course correspond with images
            masks = np.append(masks, [np.zeros([rows, cols])], axis=0)

        print(f"Calculating wavelet texture features for patient {patient}")

        # Taking the stationary (undecemated) wavelet transform of the ct images,
        # which returns a list of 8 dicts, one dict for each decomposition
        decomp = pywt.swtn(images, "coif1", level=1)[0]
        # We are interested in the HLH (i.e. dad) decomposition of the image, as it is the one
        # used for the radiomic signature in the study
        HLH = decomp["dad"]

        wavelet_images = sitk.GetImageFromArray(HLH)
        sitkmasks = sitk.GetImageFromArray(masks)

        # The size of each decomposition is the same as the original, so we can use the same
        # mask for feature calculation
        glrlm_wavelet = glrlm.RadiomicsGLRLM(wavelet_images, sitkmasks)
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
        dataframes.append(df)

    features_df = pd.concat(dataframes)

    return features_df


#%%
if __name__ == '__main__':

    csv_path = "pythondata/NSCLC Radiomics Lung1.clinical-version3-Oct 2019.csv"
    lung1_path = "C:/Users/filip/Desktop/image-data/manifest-Lung1/NSCLC-Radiomics"
    # 014, 021, 085, 095 and 194 are excluded due errors in the files provided for these patients, 128 is excluded
    # due to no segmentatiion file being provded at all (post-operative case, acounted for in study)
    disq_patients = ["LUNG1-014", "LUNG1-021", "LUNG1-085", "LUNG1-095", "LUNG1-194", "LUNG1-128"]

    # Initiating our studygroup, adding all patients, and removing those that are excluded
    lung1 = StudyGroup()
    lung1.add_all_patients(csv_path)
    remove_disqualified_patients(lung1, disq_patients)

    frame = calculate_GLCM_features(lung1, lung1_path, mute=False)

