import os
import pydicom as dicom
import re
import glob
import numpy as np
from skimage.draw import polygon
import matplotlib.pyplot as plt
import kaplanmeier as km
from patient_classes import Patient, StudyGroup, slice_viewer

testpath = r"C:\Users\filip\Desktop\haukeland_test\RS.1.2.246.352.205.4628746736953205655.4330711959846355332.dcm"


if __name__ == '__main__':
    lung1_path = r"C:\Users\filip\Desktop\radiomics_data\NSCLC-Radiomics"
    HUH_path = r"C:\Users\filip\Desktop\radiomics_data\HUH_data\1_radiomics_HUH"
    csv_path = r"C:\Users\filip\Desktop\radiomics_data\NSCLC Radiomics Lung1.clinical-version3-Oct 2019.csv"
    disq_patients = ["LUNG1-014", "LUNG1-021", "LUNG1-085", "LUNG1-095", "LUNG1-194", "LUNG1-128"]


    males_X = ["Prostate", "Lung, trachea", "Colon", "Skin, non-melanoma", "Urinary tract"]
    females_X = ["Breast", "Lung, trachea", "Colon", "Skin, non-melanoma", "Melanoma of the skin"]

    males_Y = [26.9, 9.0, 7.9, 7.2, 7.0]
    females_Y = [22.0, 10.0, 9.9, 7.2, 6.9]

    #pl = plt.bar(males_X, males_Y, color="olivedrab")
    pl = plt.bar(females_X, females_Y, color="steelblue")

    plt.grid(alpha=0.4)
    plt.xlabel("Cancer type", fontsize=12)
    plt.ylabel("Proportion of cases [%]")

    for bar in pl:
        plt.annotate(bar.get_height(),
                     xy=(bar.get_x() + 0.35, bar.get_height() + 0.2),
                     fontsize=13)

    plt.show()



    '''
    stats_df = pd.read_csv(csv_path)
    firstorder_df = pd.read_csv("pythondata/feature_files/firstorder.csv")

    stats_df = stats_df[[x not in disq_patients for x in stats_df["PatientID"]]]

    t = stats_df["Survival.time"]
    dead = stats_df["Survival.time"]
    group = firstorder_df["original_firstorder_Energy"] < firstorder_df["original_firstorder_Energy"].median()

    fo_out = km.fit(time_event=t, censoring=dead, labx=group)

    km.plot(fo_out)

    plt.show()
    '''
