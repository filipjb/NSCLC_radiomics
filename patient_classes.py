import os
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom
import re
import cv2
from skimage.draw import polygon

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
        # Survival time is measured in days, from the start of treatment
        self.deadstatus = deadstatus

        # Radiomic featurevalues calculated for the patient will be saved in each object
        self.firstorder_features = None
        self.shapebased_features = None
        self.GLCM_features = None
        self.GLRLM_features = None
        self.HLH_features = None

    def __repr__(self):
        return str(self.patientID)

    @property
    def firstorder_features(self):
        return self.__firstorder_features

    @property
    def shapebased_features(self):
        return self.__shapebased_features

    @property
    def GLCM_features(self):
        return self.__GLCM_features

    @property
    def GLRLM_features(self):
        return self.__GLRLM_features

    @property
    def HLH_features(self):
        return self.__HLH_features

    @firstorder_features.setter
    def firstorder_features(self, features):
        self.__firstorder_features = features

    @shapebased_features.setter
    def shapebased_features(self, features):
        self.__shapebased_features = features

    @GLCM_features.setter
    def GLCM_features(self, features):
        self.__GLCM_features = features

    @GLRLM_features.setter
    def GLRLM_features(self, features):
        self.__GLRLM_features = features

    @HLH_features.setter
    def HLH_features(self, features):
        self.__HLH_features = features

    # This method will take in the path to the folder containing all the the subfolders named
    # with the patient-ID: Lung1-xxx, and return the correct image array of the patient, using
    # self.patientID to find the correct directory and image array, as well as sorting the images
    # to be in the correct order accordin to slice position given in the dcm file
    def get_TCIA_images(self, path):
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
            images_dict = {}
            sorted_images = []
            # Looping through and reading the .dcm files in the folder
            for filename in os.listdir(os.getcwd()):
                dataset = dicom.dcmread(filename)
                # The location of the image slice is given by SliceLocation
                location = dataset.SliceLocation
                # Exctracting the image array from the dicom dataset and adding them to the dict
                # together with their slice location
                image_array = dataset.pixel_array
                # The RescaleIntercept is 0 if the image has SmallestImagePixelValue == 0, hence we use
                # the SmallestImagePixelValue instead to adjust the image
                if np.min(image_array) < 0:
                    # Subtract the minimum value since it is negative
                    image_array = image_array - np.min(image_array)
                images_dict.update({location: image_array})

            # Sorting the dictionary by the numerical value of the keys, i.e. slice positions
            sort = {k: v for k, v in sorted(images_dict.items(), key=lambda item: -item[0])}
            # The numpy array that will be returned to the user
            final_array = np.array(list(sort.values()))

            return final_array

    # Method to return ct images, adapted for haukeland dicomstructure
    def get_haukeland_images(self, path):
        pass

    # Method to return GTV segmentations, adapted for haukeland dicomstructure
    # TODO method for converting contour to masks is probably unreliable, rework it.
    #  There was a holefill function tried out before settlign on polygon, maybe try that
    def get_haukeland_GTV_segmentations(self, path):
        # dmcread returns a pydicom FileDataset containing entries of metadata on the patient
        seq = dicom.dcmread(path)

        # The image and segmentation data is contained in the entry tagged with StructureSetROISequence
        # It is a pydicom Sequence, where each entry is a structure that is segmented, so we loop over the structures
        # and find the one tagged with "GTV" for the tumor volume
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

        return masks

    # This method, similar to the previous, will take in the path
    # to the folder containing all the the subfolders named with
    # the patient-ID: Lung1-xxx, and return as struct containing all
    # the segmentations that are associated with the patient
    def get_TCIA_segmentations(self, path):
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
            # TODO Perhaps add some handling here for if file not found

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
    # It will return several segmentations in a dict if there are more segmentations marked
    # with the phrase "GTV"
    def get_TCIA_GTV_segmentations(self, path, return_dict=False):
        # Dict that will be returned to the user
        gtv_dict = {}
        # Fetch segmentations from the more general method
        segmentations = self.get_TCIA_segmentations(path)
        # print(f"Fetching GTV segmentations of patient: {self.patientID}")
        # Loop through the segmented volumes
        for volume in segmentations:
            # And picking out volumes tagged with "GTV"
            match = re.search("GTV", volume)
            if match:
                gtv_dict.update({volume: segmentations[volume]})
        # In case there is no GTV segmentations of the patient:
        if gtv_dict == {}:
            print(f"Error: Found no segmentations of patient {self.patientID}"
                  f" tagged with 'GTV'\n")
            quit()
        if return_dict:
            return gtv_dict
        # Since all patients in Lung1 only have a singly gtv segmentation marked with GTV-1 we can (for now) return the
        # segmentation as an array in this way
        else:
            # Array is wrong way round along the slice axis, so it is flipped to fit with CT array
            return np.flipud(gtv_dict["GTV-1"])

    # A method that will take the patient CT-images and apply outlines of the segmentations to
    # the images, returning an array of the same size to the user, needs TCIA compatible path
    def view_segmentations(self, path, window_width=550, window_height=550):

        segmentations = self.get_TCIA_segmentations(path)
        # For some reason the relative order of the two image arrays are reversed,
        # so one is flipped to account for this
        ct_images = np.flipud(self.get_TCIA_images(path))

        print(f"Showing segmentations of patient {self.patientID}")
        print(f"Segmented volumes are: {list(segmentations.keys())}")

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
        # Setting patientID as the window title of pyplot
        fig.canvas.set_window_title(self.patientID)
        # Making the window maximized when the viewer is opened
        mngr = plt.get_current_fig_manager()
        mngr.resize(window_width, window_height)

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

    def __len__(self):
        return len(self.patients)

    def index(self, item):
        return self.patients.index(item)

    # Method for adding a single patient into the group
    def add_patient(self, new_patient):
        self.patients.append(new_patient)

    # Takes in patientID as a string and removes the patient from the group
    def remove_patient(self, patientid):
        for patient in self.patients:
            if patient.patientID == patientid:
                self.patients.remove(patient)

    @staticmethod
    # A private function made for writing the two methods for adding either all patients or
    # some specific patients a little more compact
    # it takes in a line that is read from a csv file (will be a list) and returns the values
    # such that they can be used to create a patient object
    def __return_clinical_values(read_line):
        if read_line[1] == "NA":
            age = "NA"
        else:
            age = float(read_line[1])

        if read_line[2] == "NA":
            t_stage = "NA"
        else:
            t_stage = int(read_line[2])

        if read_line[3] == "NA":
            n_stage = "NA"
        else:
            n_stage = int(read_line[3])

        if read_line[4] == "NA":
            m_stage = "NA"
        else:
            m_stage = int(read_line[4])

        overall_stage = str(read_line[5])

        histology = str(read_line[6])

        gender = str(read_line[7])

        if read_line[8] == "NA":
            survival_time = "NA"
        else:
            survival_time = int(read_line[8])

        if read_line[9] == "NA":
            deadstatus = "NA"
        else:
            deadstatus = int(read_line[9])

        return age, t_stage, n_stage, m_stage, overall_stage, histology, gender, survival_time, deadstatus

    # Objective of this method is to take in the path to a .csv
    # file containing all patient data and the adding all the
    # data as patient objects into the group
    def add_all_patients(self, path):
        file = open(path, "r")
        # [1:] to skip the first line in the file, which contains the header
        for line in file.readlines()[1:]:
            line = line.split(",")
            # Extracting patient data from the line and accounting for
            # some lines containing "NA":
            patient_id = str(line[0])

            age, t, n, m, o, hist, g, st, dead = self.__return_clinical_values(line)
            patient = Patient(patient_id, age, t, n, m, o, hist, g, st, dead)
            self.patients.append(patient)
        file.close()

    # Adds specific patients to the studygroup based on a list of patientnames
    def add_specific_patients(self, path, patientnames: list):
        file = open(path, "r")
        # [1:] to skip the first line in the file, which contains the header
        for line in file.readlines()[1:]:
            line = line.split(",")
            # Extracting patient data from the line and accounting for
            # some lines containing "NA":
            patient_id = str(line[0])
            if patient_id in patientnames:
                age, t, n, m, o, hist, g, st, dead = self.__return_clinical_values(line)
                patient = Patient(patient_id, age, t, n, m, o, hist, g, st, dead)
                self.patients.append(patient)
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
    # This block is for debugging
    def slice_viewer(array):
        plt.gray()
        # Slice viewer:
        fig, ax = plt.subplots(1, 1)
        # Second argument of IndexTracker() is the array we want to
        # examine
        tracker = IndexTracker(ax, array)
        fig.canvas.mpl_connect("scroll_event", tracker.on_scroll)
        
        plt.show()

    dicom_path = "C:/Users/filip/Desktop/image-data/manifest-Lung1/NSCLC-Radiomics"
    csv_path = "pythondata/NSCLC Radiomics Lung1.clinical-version3-Oct 2019.csv"

    Lung1_group = StudyGroup()
    Lung1_group.add_all_patients(csv_path)

    patient1: Patient = Lung1_group.patients[0]

    index = 33
    image = patient1.get_TCIA_images(dicom_path)[index]
    mask = patient1.get_TCIA_segmentations(dicom_path)["Lung-Left"][index]


    plt.gray()
    plt.subplot(131), plt.imshow(image), plt.title("Image")
    plt.subplot(132), plt.imshow(mask), plt.title("Mask")
    plt.subplot(133), plt.imshow(np.multiply(image, mask)), plt.title("Segmented image")
    plt.show()




    # TODO
    #  *Segmentations and CT must be read at the same time for Haukeland images, make a single function
    #   that returns two arrays; one of CT images and one of segmentation masks
    #   -Maybe adapt TCIA functions into this single-function structure as well, if doable
    #  *Figure out have to make customizable Kaplan-Meier plots
    #  * Note that the current method for converting contour to mask might be unreliable when the
    #    segmented contour consists of more than one disjoint regions, need to rework it

