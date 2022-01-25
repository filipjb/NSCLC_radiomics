import matplotlib.pyplot as plt
import pandas as pd

huh_shape = pd.read_csv(r"feature_files\HUH_shape.csv")
lung1_shape = pd.read_csv(r"feature_files\lung1_shape.csv")
lung1_clinical = pd.read_csv(
    r"C:\Users\filip\Desktop\radiomics_data\NSCLC Radiomics Lung1.clinical-version3-Oct 2019.csv"
)
disq_patients = ["LUNG1-014", "LUNG1-021", "LUNG1-085", "LUNG1-095", "LUNG1-194", "LUNG1-128"]

for i in disq_patients:
    lung1_clinical = lung1_clinical.drop(lung1_clinical[lung1_clinical.PatientID == i].index[0])

df = pd.merge(lung1_shape, lung1_clinical, on='PatientID', how='outer')

df1 = df[df["deadstatus.event"] == 1]
df2 = df[df["deadstatus.event"] == 0]

df1.Compactness2.hist(bins=10, density=True)
df2.Compactness2.hist(bins=10, density=True)
plt.show()

