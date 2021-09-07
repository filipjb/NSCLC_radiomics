from patient_classes import Patient, StudyGroup
from matplotlib import pyplot as plt


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

    lung1 = StudyGroup()
    lung1.add_patients_from_file(csv_path)

    print_patient_statistics(lung1)

    lung1_path = "C:/Users/filip/Desktop/image-data/manifest-Lung1/NSCLC-Radiomics"

