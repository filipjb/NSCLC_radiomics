import os
import pydicom as dicom
import re
import numpy as np
from skimage.draw import polygon
import matplotlib.pyplot as plt
import pandas as pd
import kaplanmeier as km
from patient_classes import Patient, StudyGroup
import seaborn as sns

testpath = r"C:\Users\filip\Desktop\haukeland_test\RS.1.2.246.352.205.4628746736953205655.4330711959846355332.dcm"


# TODO Need to adjust mask position to patient position
def get_contour_coords(path):
    # dmcread returns a pydicom FileDataset containing many entries of metadata on the patient
    seq = dicom.dcmread(path)

    # The image and segmentation data is contained in the entry tagged with StructureSetROISequence
    # It is a pydicom Sequence, where each entry is a structure that is segmented, so we loop over the structures
    # and find the one tagge with "GTV" for the tumor volume
    for entry in seq.StructureSetROISequence:
        if re.search("GTV", entry.ROIName):
            contourNumber = int(entry.ROINumber)

    # Thus we can retrieve the pydicom Dataset corresponding to the segmented GTV, from seq.ROIContourSequence,
    # which will be a pydicom Dataset, where data on each slice is stored in a Sequence tagged with Contoursequence
    ds = seq.ROIContourSequence[contourNumber].ContourSequence

    contours = list()
    # In this Sequence, the contour coordiantes array of each entry is saved as a 1d array
    # in the tag ContourData, so we rashape the array when we retrieve it
    for n in ds:
        contourList = np.array(n.ContourData)
        # Each contoured pointed is stord sequentially; x1, y1, z1, x2, y2, z2, ..., so the array is reshaped
        # thus the contour variable contains the coordinates of the contour line around the structure
        contour = np.reshape(contourList, (len(contourList)//3, 3))
        contours.append(contour)

    # A list binary image masks that will be returned to the user
    masks = []
    # Going through each contour
    for cont in contours:
        # Creating a black image
        mask = np.zeros([512, 512])
        # Drawing a polygon at the coordinates of the contour and setting the polygon coordinates
        # in the black image to 1
        r, c = polygon(cont[:, 0], cont[:, 1], mask.shape)
        mask[r, c] = 1
        masks.append(mask)

    return masks


def get_haukeland_data(path):
    os.chdir(path)

    ct_filelist = list()
    ct_dict = dict()
    for filename in os.listdir(os.getcwd()):
        if re.search("CT", filename):
            ct_filelist.append(filename)

        if re.search("RS", filename):
            rs_filename = os.path.join(os.getcwd(), filename)

    for filename in ct_filelist:
        ct_file = dicom.dcmread(filename)
        rs_file = dicom.dcmread(rs_filename)

        ct_dict.update({ct_file["ImagePositionPatient"].value[2]: ct_file.pixel_array})
        patient_x = ct_file["ImagePositionPatient"].value[0]
        patient_y = ct_file["ImagePositionPatient"].value[1]

    # Sorting the ct dict by image slice location:
    sorted_dict = {k: v for k, v in sorted(ct_dict.items(), key=lambda item: -item[0])}
    ct_images = np.array(list(sorted_dict.values()))

    seq = dicom.dcmread(rs_filename)

    # The image and segmentation data is contained in the entry tagged with StructureSetROISequence
    # It is a pydicom Sequence, where each entry is a structure that is segmented, so we loop over the
    # structures and find the one tagge with "GTV" for the tumor volume
    for entry in seq.StructureSetROISequence:
        if re.search("GTV", entry.ROIName):
            contourNumber = int(entry.ROINumber)

    ds = seq.ROIContourSequence[contourNumber].ContourSequence
    contours = list()
    # In this Sequence, the contour coordiantes array of each entry is saved as a 1d array
    # in the tag ContourData, so we reshape the array when we retrieve it
    for n in ds:
        contourList = np.array(n.ContourData)
        # Each contoured pointed is stord sequentially; x1, y1, z1, x2, y2, z2, ..., so the array is reshaped
        # thus the contour variable contains the coordinates of the contour line around the structure
        contour = np.reshape(contourList, (len(contourList) // 3, 3))
        contours.append(contour)

    # A list binary image masks that will be returned to the user
    masks = []
    # Going through each contour
    for cont in contours:
        # Creating a black image
        mask = np.zeros([512, 512])
        # Drawing a polygon at the coordinates of the contour and setting the polygon coordinates
        # in the black image to 1
        r, c = polygon(cont[:, 0], cont[:, 1], mask.shape)
        mask[r, c] = 1
        masks.append(mask)

    ct_masks = np.array(masks)

    return ct_images, ct_masks


if __name__ == '__main__':
    dicom_path = "C:/Users/filip/Desktop/image-data/manifest-Lung1/NSCLC-Radiomics"
    csv_path = "pythondata/NSCLC Radiomics Lung1.clinical-version3-Oct 2019.csv"
    hauk_path = r"C:\Users\filip\Desktop\haukeland_test"
    disq_patients = ["LUNG1-014", "LUNG1-021", "LUNG1-085", "LUNG1-095", "LUNG1-194", "LUNG1-128"]

    Lung1_group = StudyGroup()
    Lung1_group.add_all_patients(csv_path)

    ims, masks = get_haukeland_data(hauk_path)
    print(np.shape(ims))

    plt.imshow(masks[55])
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
