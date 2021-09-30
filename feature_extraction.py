import numpy as np
import matplotlib.pyplot as plt
from patient_classes import Patient, StudyGroup
import radiomics
from radiomics import featureextractor, getTestCase, firstorder
import SimpleITK as sitk
import logging


# A simple function for converting array and associated mask into sitkImage
def convert_to_sitkImage(image_arr, mask_arr):
    return sitk.GetImageFromArray(image_arr), sitk.GetImageFromArray(mask_arr)


def calculate_firstorder_features(images, masks, featurelist=None):
    # Get the PyRadiomics logger (default log-level = INFO)
    logger = radiomics.logger
    logger.setLevel(logging.DEBUG)  # set level to DEBUG to include debug log messages in log file

    # Set up the handler to write out all log entries to a file
    handler = logging.FileHandler(filename='testLog.txt', mode='w')
    formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


    # Placeholde settings and initialization for now
    settings = {"binWidth": 25, "resampledPixelSpacing": None, "interpolator": sitk.sitkBSpline}
    # Initializing the extractor
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    # Disabling all features and then enabling only first-order features
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName(featureClass="firstorder")

    print("Calculating first-order features")
    # The feature vector returned by execute is a collections.OrderedDict
    featrue_vec = extractor.execute(images, masks)

    for featurename in featrue_vec.keys():
        print(f"Computed {featurename}: {featrue_vec[featurename]}")

    return featrue_vec


if __name__ == '__main__':
    csv_path = "pythondata/NSCLC Radiomics Lung1.clinical-version3-Oct 2019.csv"
    lung1_path = "C:/Users/filip/Desktop/image-data/manifest-Lung1/NSCLC-Radiomics"
    # 014, 021, 085 and 194 are excluded due errors in the files provided for these patients, 128 is excluded
    # due to no segmentatiion file being provded at all (post-operative case, acounted for in study)
    disq_patients = ["LUNG1-014", "LUNG1-021", "LUNG1-085", "LUNG1-194", "LUNG1-128"]

    # Initiating our studygroup, adding all patients, and removing those that are excluded
    lung1 = StudyGroup()
    lung1.add_all_patients(csv_path)
    patient1 = lung1.patients[0]

