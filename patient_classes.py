import os
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom
import matplotlib


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
    # the patient-ID: Lung1-xxx, and return the array segmentations
    # associated with this patient. Since the segmentations are contained
    # in a struct in a single .dcm file, the extraction of the array will
    # be a little different
    def return_segmentation_array(self, path):
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
            # Now the directory containg the .dcm file with the
            # segmentations, is the bottom one,
            os.chdir(os.listdir(os.getcwd())[-1])
            # There is only a single file in this directory
            filename = os.listdir(os.getcwd())[0]
            dataset = dicom.dcmread(filename)
            # The pixel array contains segmentations of the GTV, both
            # lungs and the spinal cord, as separate arrays for each
            # image slice of the patient, with the first quarter of
            # the array being the segmentations of the GTV
            segmentation_array = dataset.pixel_array
            quarter_size = int(np.shape(segmentation_array)[0]/4)
            GTV_segmentation = segmentation_array[0:quarter_size, :, :]

            return GTV_segmentation

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

    # Method will take in path to directory containg all the folders with the name
    # of the form Lung1-xxx and using the assign_dicom_files from the Patient class
    # will assign the associated file to each patient in the group
    def assign_dicoms_to_patients(self, path):
        pass

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

    patient1 = Patient("LUNG1-001", 78.7515, 2, 3, 0, "IIIb", "large cell", "male", 2165, 1)
    patient2 = Patient("LUNG1-002", 83.8001, 2, 0, 0, "I", "squamous cell carcinoma", "male", 155, 1)
    patient3 = Patient("LUNG1-003", 68.1807, 2, 3, 0, "IIIb", "large cell", "male", 256, 1)

    group = StudyGroup()
    group.add_patient(patient1)
    group.add_patient(patient2)
    group.add_patient(patient3)

    array1 = patient1.return_image_array("C:/Users/filip/Desktop/image-data/manifest-Lung1/NSCLC-Radiomics")
    array2 = patient2.return_image_array("C:/Users/filip/Desktop/image-data/manifest-Lung1/NSCLC-Radiomics")
    print("Image array has shape: ", np.shape(array1))

    segmentation1 = patient1.return_segmentation_array(
        "C:/Users/filip/Desktop/image-data/manifest-Lung1/NSCLC-Radiomics"
    )

    segmentation2 = patient2.return_segmentation_array(
        "C:/Users/filip/Desktop/image-data/manifest-Lung1/NSCLC-Radiomics"
    )

    print("Segmentation array has shape: ", np.shape(segmentation1))

    index = 80

    plt.gray()
    plt.subplot(1, 3, 1), plt.imshow(segmentation1[index, :, :])
    plt.subplot(1, 3, 2), plt.imshow(array1[index, :, :])
    plt.subplot(1, 3, 3), plt.imshow(
        np.multiply(segmentation1[index, :, :], array1[index, :, :])
    )
    plt.show()
