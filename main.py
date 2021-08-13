from patient_classes import Patient, StudyGroup
from matplotlib import pyplot as plt

if __name__ == '__main__':

    csv_path = "pythondata/NSCLC Radiomics Lung1.clinical-version3-Oct 2019.csv"

    Lung1_group = StudyGroup()
    Lung1_group.add_patients_from_file(csv_path)

    T = Lung1_group.relative_frequency_Tstages()
    N = Lung1_group.relative_frequency_Nstages()
    TNM = Lung1_group.relative_frequency_TNM()

    '''
    Patient statistics:
    
    print("Males: ", Lung1_group.relative_frequency_males())
    print("Females: ", Lung1_group.relative_frequency_females())

    print("Mean age", Lung1_group.mean_age())
    print("Age range", Lung1_group.age_range())

    print("\n")

    print("T1", T[0])
    print("T2", T[1])
    print("T3", T[2])
    print("T4", T[3])
    print("Tx", T[4])
    print("T Sum", sum(T))

    print("\n")

    print("N0", N[0])
    print("N1", N[1])
    print("N2", N[2])
    print("N3", N[3])
    print("Nx", N[4])
    print("N Sum", sum(N))

    print("\n")

    print(TNM, sum(TNM))

    '''

    path = "C:/Users/filip/Desktop/image-data/manifest-Lung1/NSCLC-Radiomics"

    print(Lung1_group[55].return_segmentations(path))

