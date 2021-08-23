import os
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom
import re
import cv2

from slice_viewer import IndexTracker


# The methods to assign dicoms to patients makes assumption on the file
# structure that is downloaded from TCIA using descriptive directory
# structure, methods must be rewritten if folders are provided in
# different formats, may come back to this later to fine more universally
# applicable method
class Patient:
    # Objective is to input data from the csv file into objects of the class Patient,
    # with the Patient objects being collected into an instance of the StudyGroup class
    def __init__(self, patientID, age, T_stage, N_stage, M_stage, overall_stage,
                 histology, gender, survival_time, deadstatus):
        self.patientID = patientID
        self.age = age
        self.T_stage = T_stage
        self.N_stage = N_stage
        self.M_stage = M_stage
        self.overall_stage = overall_stage
        self.histology = histology
        self.gender = gender
        self.survival_time = survival_time
        self.deadstatus = deadstatus
        self.dicoms = []

    # This method will take in the path to the folder containing
    # all the the subfolders named with the patient-ID: Lung1-xxx,
    # and return the image correct image array of the patient, using
    # self.patientID to find the correct directory and image array
    def return_image_array(self, path):
        # Changing directory to the folder contatining patient
        # subfolders
        os.chdir(path)
        # Managing if the user has provided a faulty or wrong path
        try:
            # Entering the directory with the name of the patient
            os.chdir(str(self.patientID))
            # If no file is found with the patientID in the provided
            # path an error is returned to the user, and the process
            # is terminated
        except FileNotFoundError as e:
            print(f"\nError: The specified path is not a directory containing"
                  f" the expected patient-ID: {self.patientID}")
        else:
            # Entering the top folder in this directory
            os.chdir(os.listdir(os.getcwd())[0])
            # Again entering the top folder inside this directory, which contains
            # all the dicom files for the patient
            os.chdir(os.listdir(os.getcwd())[0])
            images = []
            # Looping through and reading the .dcm files in the folder
            for filename in os.listdir(os.getcwd()):
                dataset = dicom.dcmread(filename)
                # Exctracting the image array from the dicom dataset
                image_array = dataset.pixel_array
                # And appending it to the total array of all images of the patient
                images.append(image_array)

            return np.array(images)

    # This method, similar to the previous, will take in the path
    # to the folder containing all the the subfolders named with
    # the patient-ID: Lung1-xxx, and return as struct containing all
    # the segmentations that are associated with the patient
    def return_segmentations(self, path):
        # Changing directory to the folder contatining patient
        # subfolders
        os.chdir(path)
        # Managing if the user has provided a faulty or wrong path
        try:
            # Entering the directory with the name of the patient
            os.chdir(str(self.patientID))
            # If no file is found with the patientID in the provided
            # path an error is returned to the user, and the process
            # is terminated
        except FileNotFoundError as e:
            print(f"\nError: The specified path is not a directory containing"
                  f" the expected patient-ID: {self.patientID}")
        else:
            # Entering the top folder in this directory
            os.chdir(os.listdir(os.getcwd())[0])
            # Now the directory containing the .dcm file with the
            # segmentations, contains the string "Segmentation", so
            # we lopp throught the directory names in cwd:
            for dirname in os.listdir(os.getcwd()):
                # If a directory contains the string, we choose it
                if re.search("Segmentation", dirname):
                    os.chdir(os.path.join(os.getcwd(), dirname))
            # Perhaps add some handling here for if file not found

            # In this directory there is a single.dcm file containg
            # all the segmentations for the patient
            filename = os.listdir(os.getcwd())[0]
            dataset = dicom.dcmread(filename)
            # Check that the dataset is marked with the correct patientID:
            if dataset["PatientID"].value == self.patientID:
                pass
            else:
                print(f"Error: Patient object ID ({self.patientID}), does "
                      f"not correspond with the patient ID in the provided"
                      f"dataset ({dataset['PatientID'].value})")
                quit()

            # Constructing a dict that will contain the tags for the organs that
            # are segmented, together with their respective image matrices:
            segmentation_dict = {}
            for entry in dataset["SegmentSequence"]:
                segmentation_dict.update({entry["SegmentDescription"].value: None})

            # The array containing all segmentations in order:
            total_array = dataset.pixel_array
            length, rows, cols = np.shape(total_array)

            # The array is split into equal sections, each section being the number of
            # images in the total segmentation array divided by the number of segmentations
            split_array = total_array.reshape(len(segmentation_dict), -1, rows, cols)

            for keyword in segmentation_dict:
                # The segmentations are added to the segmentation_dict in the order they
                # appear in the dict
                index = list(segmentation_dict.keys()).index(keyword)
                segmentation_dict[keyword] = split_array[index, :, :, :]

            return segmentation_dict

    # A method for returning only the GTV segmentation for when radiomics will be computed
    def return_GTV_segmentation(self, path):
        pass

    # A method that will take the patient CT-images and apply outlines of the segmentations to
    # the images, returning an array of the same size to the user
    def view_segmentations(self, path):

        segmentations = self.return_segmentations(path)
        # For some reason the relative order of the two image arrays are reversed,
        # so one is flipped to account for this
        ct_images = np.flipud(self.return_image_array(path))

        # The array that will be returned to the user
        ct_rgb_images = []
        # Looping through the ct-slices of the patient
        for image in ct_images:
            # Each image is uint8 normalized in order to be converted to rgb
            image = cv2.normalize(
                image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            # Each image is converted into rgb such that coloured contours can be displayed
            # on them
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            # The image is appended to the array that will be returned to the user, creating
            # the same ct-array as earlier only that the images are in rgb
            ct_rgb_images.append(image)
        # converting rgb ct-list into numpy array
        ct_rgb_images = np.array(ct_rgb_images)

        # Looping through each organ volume segmentation in the segmentation dict
        for volume in segmentations:
            # segmentations[volume] will be the array of segmentation images of the specific organ,
            # and is a binary image which we easily can find the contour of
            bw_array = segmentations[volume]
            # Creating a random rgb colour which will colour the contour of this volume segmentation
            rgb = [np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)]

            # Looping through all the images in the segmentation array
            for i in range(len(bw_array)):
                # Picking an image in the segmentation array, and the corresponding ct-image
                bw_image = bw_array[i, :, :]
                image = ct_rgb_images[i, :, :, :]
                # Finding the contours on the binary image:
                contours, _ = cv2.findContours(bw_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                # The contours are drawn on the corresponding ct image, and the change is made to the
                # rgb-array
                ct_rgb_images[i, :, :, :] = cv2.drawContours(image, contours, -1, rgb, 2)
        # Viewing the result in the slice-viewer:
        # Slice viewer:
        fig, ax = plt.subplots(1, 1)
        # Second argument of IndexTracker() is the array we want to
        # examine
        tracker = IndexTracker(ax, ct_rgb_images)
        fig.canvas.mpl_connect("scroll_event", tracker.on_scroll)
        plt.show()

    def __str__(self):
        return f"{self.patientID}"


class StudyGroup:

    def __init__(self):
        self.patients = []

    def __str__(self):
        result = "Group: \n"
        for patient in self.patients:
            result += str(patient) + "\n"
        return result

    def __getitem__(self, item):
        return self.patients[item]

    # Method for adding a single patient into the group
    def add_patient(self, new_patient):
        self.patients.append(new_patient)

    # Objective of this method is to take in the path to a .csv
    # file containing all patient data and the adding all the
    # data as patient objects into the group
    def add_patients_from_file(self, path):
        file = open(path, "r")
        # [1:] to skip the first line in the file, which contains the header
        for line in file.readlines()[1:]:
            line = line.split(",")
            # Extracting patient data from the line and accounting for
            # some lines containing "NA":
            patient_id = str(line[0])

            if line[1] == "NA":
                age = "NA"
            else:
                age = float(line[1])

            if line[2] == "NA":
                t_stage = "NA"
            else:
                t_stage = int(line[2])

            if line[3] == "NA":
                n_stage = "NA"
            else:
                n_stage = int(line[3])

            if line[4] == "NA":
                m_stage = "NA"
            else:
                m_stage = int(line[4])

            overall_stage = str(line[5])

            histology = str(line[6])

            gender = str(line[7])

            if line[8] == "NA":
                survival_time = "NA"
            else:
                survival_time = int(line[8])

            if line[9] == "NA":
                deadstatus = "NA"
            else:
                deadstatus = int(line[9])

            # Creating a patient object from the data extracted from
            # the current line
            patient = Patient(patient_id, age, t_stage, n_stage, m_stage, overall_stage,
                              histology, gender, survival_time, deadstatus)
            # And adding the patient to the studygroup
            self.patients.append(patient)
        # Closing the file after all patients are addded
        file.close()

    def mean_age(self):
        result = 0
        invalid = 0
        for patient in self.patients:
            # Pasients who have age "NA" are accounted for by
            # removing them from the dataset
            if patient.age == "NA":
                invalid += 1
            else:
                result += np.floor(patient.age)
        return result/(self.size() - invalid)

    def age_range(self):
        ages = []
        for patient in self.patients:
            if patient.age != "NA":
                ages.append(patient.age)
        return [min(ages), max(ages)]

    def size(self):
        return len(self.patients)

    # The two following methods goes throught the patients and
    # counts the number of males and females, respectivly. They
    # then return the relative frequency of males and females
    # in the StudyGroup
    def relative_frequency_males(self):
        result = 0
        invalid = 0
        for patient in self.patients:
            # Accounting for i gender is NA
            if patient.gender == "NA":
                invalid += 1
            elif patient.gender == "male":
                result += 1
        return (result/(self.size() - invalid))*100

    def relative_frequency_females(self):
        result = 0
        invalid = 0
        for patient in self.patients:
            # Account for if gender is NA
            if patient.gender == "NA":
                invalid += 1
            elif patient.gender == "female":
                result += 1
        return (result/self.size())*100

    # This method will return the relative frequencies of all the T
    # stages in a list of the form [T1, T2, T3, T4, Tx]
    def relative_frequency_Tstages(self):
        T1 = 0
        T2 = 0
        T3 = 0
        T4 = 0
        Tx = 0
        for patient in self.patients:
            # Med antakelse at NA refererer til Tx stadiet
            if patient.T_stage == 1:
                T1 += 1
            elif patient.T_stage == 2:
                T2 += 1
            elif patient.T_stage == 3:
                T3 += 1
            elif patient.T_stage == 4:
                T4 += 1
            # Patient stages not accounted for by the previous metrics are
            # counted in the group Tx
            else:
                Tx += 1
        return [
            T1 / self.size()*100, T2 / self.size()*100,
            T3 / self.size()*100, T4 / self.size()*100,
            Tx / self.size()*100
        ]

    def relative_frequency_Nstages(self):
        N0 = 0
        N1 = 0
        N2 = 0
        N3 = 0
        Nx = 0
        for patient in self.patients:
            if patient.N_stage == 0:
                N0 += 1
            elif patient.N_stage == 1:
                N1 += 1
            elif patient.N_stage == 2:
                N2 += 1
            elif patient.N_stage == 3:
                N3 += 1
            # Patient stages not accounted for by the previous metrics are
            # counted in the group Nx
            else:
                Nx += 1
        return [
            N0 / self.size() * 100, N1 / self.size() * 100,
            N2 / self.size() * 100, N3 / self.size() * 100,
            Nx / self.size() * 100
        ]

    def relative_frequency_TNM(self):
        stage1 = 0
        stage2 = 0
        stage3a = 0
        stage3b = 0
        invalid = 0
        for patient in self.patients:
            if patient.overall_stage == "I":
                stage1 += 1
            elif patient.overall_stage == "II":
                stage2 += 1
            elif patient.overall_stage == "IIIa":
                stage3a += 1
            elif patient.overall_stage == "IIIb":
                stage3b += 1
            # There is a single patient which has "NA" overall stage, and
            # coincidentally has a T stage of 5, which seems to have netted the
            # patient of being placed in the overall stage IIIb when the statistics
            # have been calculated
            else:
                stage3b += 1
        return [
            stage1 * 100 / (self.size() - invalid), stage2 * 100 / (self.size() - invalid),
            stage3a * 100 / (self.size() - invalid), stage3b * 100 / (self.size() - invalid)
        ]


if __name__ == '__main__':

    dicom_path = "C:/Users/filip/Desktop/image-data/manifest-Lung1/NSCLC-Radiomics"
    csv_path = "pythondata/NSCLC Radiomics Lung1.clinical-version3-Oct 2019.csv"

    Lung1_group = StudyGroup()
    Lung1_group.add_patients_from_file(csv_path)

    patient = Lung1_group[231]

    patient.view_segmentations(dicom_path)

    '''
    
    # Slice viewer:
    fig, ax = plt.subplots(1, 1)
    # Second argument of IndexTracker() is the array we want to
    # examine
    tracker = IndexTracker(ax, a)
    fig.canvas.mpl_connect("scroll_event", tracker.on_scroll)
    plt.show()
    '''
