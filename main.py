from patient_classes import Patient, StudyGroup
from matplotlib import pyplot as plt


# Given a list of patient IDs the function removes the patients from the given studgroup
def remove_disqualified_patients(group: StudyGroup, patients: list):
    for name in patients:
        group.remove_patient(name)


def print_patient_statistics(group):

    T = group.relative_frequency_Tstages()
    N = group.relative_frequency_Nstages()
    TNM = group.relative_frequency_TNM()

    print("Males: ", group.relative_frequency_males())
    print("Females: ", group.relative_frequency_females())

    print("Mean age", group.mean_age())
    print("Age range", group.age_range())

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


if __name__ == '__main__':

    csv_path = "pythondata/NSCLC Radiomics Lung1.clinical-version3-Oct 2019.csv"
    lung1_path = "C:/Users/filip/Desktop/image-data/manifest-Lung1/NSCLC-Radiomics"
    # 014, 021, 085 and 194 are excluded due errors in the files provided for these patients, 128 is excluded
    # due to no segmentatiion file being provded at all (post-operative case, acounted for in study)
    disq_patients = ["LUNG1-014", "LUNG1-021", "LUNG1-085", "LUNG1-194", "LUNG1-128"]

    # Initiating our studygroup, adding all patients, and removing those that are excluded
    lung1 = StudyGroup()
    lung1.add_all_patients(csv_path)
    remove_disqualified_patients(lung1, disq_patients)

    # Creating a separate group object for the patients that are excluded, such that we can calculate
    # their statistics
    excluded_lung1 = StudyGroup()
    excluded_lung1.add_specific_patients(csv_path, disq_patients)

