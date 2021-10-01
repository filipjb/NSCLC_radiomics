import numpy as np
import matplotlib.pyplot as plt
from patient_classes import Patient, StudyGroup
from radiomics import featureextractor, getTestCase, firstorder
import SimpleITK as sitk


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

    extractor.enableFeatureClassByName(featureClass="firstorder")

    print(f"\nCalculating first-order features for patient {patient}")
    # The feature vector returned by execute is a collections.OrderedDict
    feature_dict = extractor.execute(images, masks)
    if not mute:
        for featurename in feature_dict.keys():
            print(f"Computed {featurename}: {feature_dict[featurename]}")

    patient.firstorder_features = feature_dict


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

    calculate_firstorder_features(patient1, lung1_path)
    print(patient1.firstorder_features)
