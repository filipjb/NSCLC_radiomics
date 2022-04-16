import os
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom
import re
import cv2
import glob
from skimage.draw import polygon
from slice_viewer import IndexTracker


class Patient:

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

        # If needed, feature values can be assigned to the patient objects
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

    # The methods to load dicoms from patients make assumption on the TCIA or HUH individual
    # file structures to explicitly navigate the directories
    def get_TCIA_images(self, path):
        os.chdir(path)
        # Managing if the user has provided a faulty or wrong path
        try:
            os.chdir(str(self.patientID))
        except FileNotFoundError as e:
            print(e, "\n")
            print(f"\nError: The specified path is not a directory containing"
                  f" the expected patient-ID: {self.patientID}")
        else:
            # Navigating TCIA file structure
            os.chdir(os.listdir(os.getcwd())[0])
            os.chdir(os.listdir(os.getcwd())[0])
            images_dict = {}
            # Retrieving images
            for filename in os.listdir(os.getcwd()):
                dataset = dicom.dcmread(filename)
                location = dataset.SliceLocation
                image_array = dataset.pixel_array
                # Fixing images not being consistent across patients with pixel intensity range
                if np.min(image_array) < 0:
                    image_array = image_array - np.min(image_array)
                images_dict.update({location: image_array})

            # Sorting the dictionary by the numerical value of the keys, i.e. slice positions
            sort = {k: v for k, v in sorted(images_dict.items(), key=lambda item: -item[0])}
            final_array = np.array(list(sort.values()))

            return final_array

    def get_haukeland_data(self, path, structure="GTVp"):
        os.chdir(os.path.join(path, str(self.patientID)))
        ct_dict = dict()
        masks = dict()

        ct_filelist = glob.glob(os.path.join(os.getcwd(), r"CT*.dcm"))
        rs_filename = glob.glob(os.path.join(os.getcwd(), r"RS*.dcm"))
        if not ct_filelist or not rs_filename:
            raise FileNotFoundError

        for n in range(len(ct_filelist)):

            # Handling CT-images
            ct = dicom.dcmread(ct_filelist[n])
            ct_dict.update({ct.ImagePositionPatient[2]: ct.pixel_array})

            # Handling segmentations
            # Extracting patient position from ct dicom
            patient_x = ct.ImagePositionPatient[0]
            patient_y = ct.ImagePositionPatient[1]
            patient_z = ct.ImagePositionPatient[2]
            ps = ct.PixelSpacing[0]

            seq = dicom.dcmread(rs_filename[0])
            # Finding the contournumber of the selected structure, such that we can extract it from ROIContourSequence
            structureNames = [seq.StructureSetROISequence[i].ROIName for i in range(len(seq.StructureSetROISequence))]
            contourNumber = [i for i, item in enumerate(structureNames) if re.search(structure, item)][0]

            ds = seq.ROIContourSequence[contourNumber].ContourSequence

            totalMask = np.zeros([ct.pixel_array.shape[0], ct.pixel_array.shape[1]])
            for element in ds:
                # If the UID of the contour matches the UID of the sequence, we retrieve the contour:
                if element.ContourImageSequence[0].ReferencedSOPInstanceUID == ct.SOPInstanceUID:
                    contour = np.array(element.ContourData)
                    # Each contoured point is stored sequentially; x1, y1, z1, x2, y2, z2, ...,
                    # so the array is reshaped thus the contour variable contains the coordinates of the
                    # contour line around the structure
                    contour = np.reshape(contour, (len(contour) // 3, 3))
                    # Make the contour into a mask:
                    contourMask = np.zeros([ct.pixel_array.shape[0], ct.pixel_array.shape[1]])
                    r, c = polygon((contour[:, 0] - patient_x) / ps, (contour[:, 1] - patient_y) / ps,
                                   contourMask.shape)
                    contourMask[r, c] = 1
                    totalMask += np.fliplr(np.rot90(contourMask, axes=(1, 0)))

            masks.update({patient_z: totalMask > 0})

        # Sorting the ct dict by image slice position:
        sorted_dict = {k: v for k, v in sorted(ct_dict.items(), key=lambda item: -item[0])}
        ct_images = np.array(list(sorted_dict.values()))

        # Sorting patient contours by slice position:
        sorted_contours = {k: v for k, v in sorted(masks.items(), key=lambda item: -item[0])}
        ct_masks = np.array(list(sorted_contours.values())).astype(np.uint8)

        return ct_images, ct_masks

    def get_TCIA_segmentations(self, path):
        os.chdir(path)
        try:
            os.chdir(str(self.patientID))
        except FileNotFoundError as e:
            print(f"\nError: The specified path is not a directory containing"
                  f" the expected patient-ID: {self.patientID}")
        else:
            os.chdir(os.listdir(os.getcwd())[0])
            for dirname in os.listdir(os.getcwd()):
                if re.search("Segmentation", dirname):
                    os.chdir(os.path.join(os.getcwd(), dirname))
            # TODO Perhaps add some handling here for if file not found

            filename = os.listdir(os.getcwd())[0]
            dataset = dicom.dcmread(filename)
            if dataset["PatientID"].value == self.patientID:
                pass
            else:
                print(f"Error: Patient object ID ({self.patientID}), does "
                      f"not correspond with the patient ID in the provided"
                      f"dataset ({dataset['PatientID'].value})")
                quit()

            segmentation_dict = {}
            for entry in dataset["SegmentSequence"]:
                segmentation_dict.update({entry["SegmentDescription"].value: None})

            total_array = dataset.pixel_array
            length, rows, cols = np.shape(total_array)

            # The array is split into equal sections, each section being the number of
            # images in the total segmentation array divided by the number of segmentations
            split_array = total_array.reshape(len(segmentation_dict), -1, rows, cols)

            for keyword in segmentation_dict:
                index = list(segmentation_dict.keys()).index(keyword)
                # The assigned array is flipped along the slice axis to correspond with CT image order
                segmentation_dict[keyword] = np.flipud(split_array[index, :, :, :])

            return segmentation_dict

    # A method for returning only the GTV segmentations, for calculating radiomics
    def get_TCIA_GTV_segmentations(self, path, return_dict=False):
        gtv_dict = {}
        segmentations = self.get_TCIA_segmentations(path)
        for volume in segmentations:
            match = re.search("GTV", volume)
            if match:
                gtv_dict.update({volume: segmentations[volume]})
        if gtv_dict == {}:
            print(f"Error: Found no segmentations of patient {self.patientID}"
                  f" tagged with 'GTV'\n")
            quit()
        if return_dict:
            return gtv_dict
        else:
            return gtv_dict["GTV-1"]

    # Method for viewing delineations on top of the CT images
    def view_segmentations(self, path, pathtype="TCIA", window_width=550, window_height=550):

        if pathtype == "TCIA":
            segmentations = self.get_TCIA_segmentations(path)
            ct_images = self.get_TCIA_images(path)
            print(f"Showing segmentations of patient {self.patientID}")
            print(f"Segmented volumes are: {list(segmentations.keys())}")

            ct_rgb_images = []
            for image in ct_images:
                image = cv2.normalize(
                    image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
                )
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                ct_rgb_images.append(image)
            ct_rgb_images = np.array(ct_rgb_images)

            for volume in segmentations:
                bw_array = segmentations[volume]
                rgb = [np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)]
                for i in range(len(bw_array)):
                    bw_image = bw_array[i, :, :]
                    image = ct_rgb_images[i, :, :, :]
                    contours, _ = cv2.findContours(bw_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    ct_rgb_images[i, :, :, :] = cv2.drawContours(image, contours, -1, rgb, 2)

            # Viewing the result in the slice-viewer
            fig, ax = plt.subplots(1, 1)
            tracker = IndexTracker(ax, ct_rgb_images)
            fig.canvas.mpl_connect("scroll_event", tracker.on_scroll)
            fig.canvas.set_window_title(self.patientID)
            mngr = plt.get_current_fig_manager()
            mngr.resize(window_width, window_height)

            plt.show()

        elif pathtype == "HUH":
            ct_images, segmentations = self.get_haukeland_data(path, structure="GTV")
            segmentations = segmentations.astype(np.uint8)
            ct_rgb_images = []
            for image in ct_images:
                image = cv2.normalize(
                    image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
                )
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                ct_rgb_images.append(image)
            ct_rgb_images = np.array(ct_rgb_images)

            rgb = [np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)]

            for i in range(len(segmentations)):
                bw_image = segmentations[i, :, :]
                image = ct_rgb_images[i, :, :, :]
                contours, _ = cv2.findContours(bw_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                ct_rgb_images[i, :, :, :] = cv2.drawContours(image, contours, -1, rgb, 2)

            fig, ax = plt.subplots(1, 1)
            tracker = IndexTracker(ax, ct_rgb_images)
            fig.canvas.mpl_connect("scroll_event", tracker.on_scroll)
            fig.canvas.set_window_title(self.patientID)
            mngr = plt.get_current_fig_manager()
            mngr.resize(window_width, window_height)

            plt.show()

        else:
            print("Error: Unrecognized pathtype")
            quit()

    def __str__(self):
        return f"{self.patientID}"


class StudyGroup:

    def __init__(self, groupID):
        self.patients = []
        self.groupID = groupID

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

    def add_patient(self, new_patient):
        self.patients.append(new_patient)

    def remove_patient(self, patientid):
        for patient in self.patients:
            if patient.patientID == patientid:
                self.patients.remove(patient)

    def remove_multiple_patients(self, patients: list):
        for name in patients:
            self.remove_patient(name)

    @staticmethod
    # A private function for making other methods more compact
    def __return_clinical_values(read_line):
        if read_line[1] == "NA" or read_line[1] == "":
            age = "NA"
        else:
            age = float(read_line[1])

        if read_line[2] == "NA" or read_line[2] == "":
            t_stage = "NA"
        else:
            t_stage = int(read_line[2])

        if read_line[3] == "NA" or read_line[3] == "":
            n_stage = "NA"
        else:
            n_stage = int(read_line[3])

        if read_line[4] == "NA" or read_line[4] == "":
            m_stage = "NA"
        else:
            m_stage = int(read_line[4])

        overall_stage = str(read_line[5])

        histology = str(read_line[6])

        gender = str(read_line[7])

        if read_line[8] == "NA" or read_line[8] == "":
            survival_time = "NA"
        else:
            survival_time = int(read_line[8])

        if read_line[9] == "NA" or read_line[9] == "":
            deadstatus = "NA"
        else:
            deadstatus = int(read_line[9])

        return age, t_stage, n_stage, m_stage, overall_stage, histology, gender, survival_time, deadstatus

    # A method that can add HUH patients to the group using the directory containing the image data
    # instead of csv
    def add_HUH_patients(self, path):
        os.chdir(path)
        for dirname in os.listdir(os.getcwd()):
            patient = Patient(dirname, None, None, None, None, None, None, None, None, None)
            self.patients.append(patient)

    def add_all_patients(self, path, pathtype="TCIA"):
        file = open(path, "r")
        for line in file.readlines()[1:]:
            line = line.strip()
            if pathtype == "TCIA":
                line = line.split(",")
            elif pathtype == "HUH":
                line = line.split(";")
            patient_id = str(line[0])
            age, t, n, m, o, hist, g, st, dead = self.__return_clinical_values(line)
            patient = Patient(patient_id, age, t, n, m, o, hist, g, st, dead)
            self.patients.append(patient)
        file.close()

    def add_specific_patients(self, path, patientnames: list):
        file = open(path, "r")
        for line in file.readlines()[1:]:
            line = line.split(",")

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

    def relative_frequency_males(self):
        result = 0
        invalid = 0
        for patient in self.patients:
            if patient.gender == "NA":
                invalid += 1
            elif patient.gender == "male":
                result += 1
        return (result/(self.size() - invalid))*100

    def relative_frequency_females(self):
        result = 0
        invalid = 0
        for patient in self.patients:
            if patient.gender == "NA":
                invalid += 1
            elif patient.gender == "female":
                result += 1
        return (result/self.size())*100

    def relative_frequency_Tstages(self):
        T1 = 0
        T2 = 0
        T3 = 0
        T4 = 0
        Tx = 0
        for patient in self.patients:
            if patient.T_stage == 1:
                T1 += 1
            elif patient.T_stage == 2:
                T2 += 1
            elif patient.T_stage == 3:
                T3 += 1
            elif patient.T_stage == 4:
                T4 += 1
            # NA entries are assumed to refer to Tx stages
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
            # NA entries are assumed to refer to Nx stages
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
            elif patient.overall_stage == "IIIa" or patient.overall_stage == "IIIA":
                stage3a += 1
            elif patient.overall_stage == "IIIb" or patient.overall_stage == "IIIB":
                stage3b += 1
            # There is a single patient which has "NA" overall stage in Lung1, and
            # coincidentally has a T stage of 5, which seems to have netted the
            # patient of being placed in the overall stage IIIb when the statistics
            # have been calculated in Aerts et al.
            else:
                stage3b += 1
        return [
            stage1 * 100 / (self.size() - invalid), stage2 * 100 / (self.size() - invalid),
            stage3a * 100 / (self.size() - invalid), stage3b * 100 / (self.size() - invalid)
        ]

    def print_statistics(self):
        T = self.relative_frequency_Tstages()
        N = self.relative_frequency_Nstages()
        TNM = self.relative_frequency_TNM()

        print("Males: ", self.relative_frequency_males())
        print("Females: ", self.relative_frequency_females())

        print("Mean age", self.mean_age())
        print("Age range", self.age_range())
        print()
        print("T1:", T[0])
        print("T2:", T[1])
        print("T3:", T[2])
        print("T4:", T[3])
        print("Tx:", T[4])
        print("T Sum:", sum(T))
        print()
        print("N0:", N[0])
        print("N1:", N[1])
        print("N2:", N[2])
        print("N3:", N[3])
        print("Nx:", N[4])
        print("N Sum:", sum(N))
        print()
        print("TNM:", TNM)
        print("TNM sum:", sum(TNM))


# This block is for debugging
if __name__ == '__main__':

    lung1_path = r"C:\Users\filip\Downloads\radiomics_data\NSCLC-Radiomics"
    huh_path = r"C:\Users\filip\Downloads\radiomics_data\HUH_data"
    csv_path = r"NSCLC Radiomics Lung1.clinical-version3-Oct 2019.csv"
    lung1_disq = ["LUNG1-014", "LUNG1-021", "LUNG1-085", "LUNG1-095", "LUNG1-128", "LUNG1-194"]
    huh_disq = ["26_radiomics_HUH", "27_radiomics_HUH", "28_radiomics_HUH"]

    lung1: StudyGroup = StudyGroup("lung1")
    lung1.add_all_patients(csv_path, pathtype="TCIA")
    lung1.remove_multiple_patients(lung1_disq)

    huh = StudyGroup("huh")
    huh.add_all_patients("HUH_clinical.csv", pathtype="HUH")
    huh.remove_multiple_patients(huh_disq)

    lung1.print_statistics()


