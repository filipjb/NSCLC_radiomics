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


# TODO Implement this into Patient Class when file-structure is known for whole collection, such that
#  patient data can be retrieved based on Patient-ID
def get_haukeland_data(path, structure="GTV"):
    os.chdir(path)
    ct_dict = dict()
    masks = dict()

    ct_filelist = glob.glob(os.path.join(os.getcwd(), r"CT*.dcm"))
    rs_filename = glob.glob(os.path.join(os.getcwd(), r"RS*.dcm"))
    if not ct_filelist or not rs_filename:
        raise FileNotFoundError

    for n in range(len(ct_filelist)):

        # ------------ Handling CT-images -------------- #
        ct = dicom.dcmread(ct_filelist[n])
        # Adding images to dict, paired with their position along the z-axis
        ct_dict.update({ct.ImagePositionPatient[2]: ct.pixel_array})

        # ------------- Handling segmentations -------------- #
        # Extracting patient position from ct dicom
        patient_x = ct.ImagePositionPatient[0]
        patient_y = ct.ImagePositionPatient[1]
        patient_z = ct.ImagePositionPatient[2]
        ps = ct.PixelSpacing[0]

        seq = dicom.dcmread(rs_filename[0])     # The dicomfile for the segmented structures
        # Finding the contournumber of the selected structure, such that we can extract it from ROIContourSequence
        structureNames = [seq.StructureSetROISequence[i].ROIName for i in range(len(seq.StructureSetROISequence))]
        contourNumber = [i for i, item in enumerate(structureNames) if re.search(structure, item)][0]

        # The contoursequence of the structure we have chosen
        ds = seq.ROIContourSequence[contourNumber].ContourSequence

        totalMask = np.zeros([ct.pixel_array.shape[0], ct.pixel_array.shape[1]])
        for element in ds:
            # If the UID of the contour matches the UID of the sequence, we retrieve the contour:
            if element.ContourImageSequence[0].ReferencedSOPInstanceUID == ct.SOPInstanceUID:
                contour = np.array(element.ContourData)
                # Each contoured pointed is stored sequentially; x1, y1, z1, x2, y2, z2, ..., so the array is reshaped
                # thus the contour variable contains the coordinates of the contour line around the structure
                contour = np.reshape(contour, (len(contour) // 3, 3))
                # Make the contour into a mask:
                contourMask = np.zeros([ct.pixel_array.shape[0], ct.pixel_array.shape[1]])
                r, c = polygon((contour[:, 0] - patient_x) / ps, (contour[:, 1] - patient_y) / ps, contourMask.shape)
                contourMask[r, c] = 1
                totalMask += np.fliplr(np.rot90(contourMask, axes=(1, 0)))

        masks.update({patient_z: totalMask > 0})

    # Sorting the ct dict by image slice position:
    sorted_dict = {k: v for k, v in sorted(ct_dict.items(), key=lambda item: -item[0])}
    ct_images = np.array(list(sorted_dict.values()))

    # Sorting patient contours by slice position:
    sorted_contours = {k: v for k, v in sorted(masks.items(), key=lambda item: -item[0])}
    ct_masks = np.array(list(sorted_contours.values()))

    return ct_images, ct_masks


if __name__ == '__main__':
    dicom_path = "C:/Users/filip/Desktop/image-data/manifest-Lung1/NSCLC-Radiomics"
    csv_path = "pythondata/NSCLC Radiomics Lung1.clinical-version3-Oct 2019.csv"
    hauk_path = r"C:\Users\filip\Desktop\haukeland_test"
    disq_patients = ["LUNG1-014", "LUNG1-021", "LUNG1-085", "LUNG1-095", "LUNG1-194", "LUNG1-128"]

    Lung1_group = StudyGroup()
    #Lung1_group.add_all_patients(csv_path)

    ims, masks = get_haukeland_data(hauk_path, "Lungs")

    slice_viewer(np.multiply(masks, ims))
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
