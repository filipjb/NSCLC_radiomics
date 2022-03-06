import matplotlib.pyplot as plt
import pandas as pd
import kaplanmeier as km

huh_shape = pd.read_csv(r"feature_files\HUH_shape.csv")
lung1_shape = pd.read_csv(r"feature_files\lung1_shape.csv")
lung1_clinical = pd.read_csv(
    r"C:\Users\filip\Desktop\radiomics_data\NSCLC Radiomics Lung1.clinical-version3-Oct 2019.csv"
)
disq_patients = ["LUNG1-014", "LUNG1-021", "LUNG1-085", "LUNG1-095", "LUNG1-194", "LUNG1-128"]

# Removing the patients in Lung1 that were disqualified
for i in disq_patients:
    lung1_clinical = lung1_clinical.drop(lung1_clinical[lung1_clinical.PatientID == i].index[0])

df = pd.merge(lung1_shape, lung1_clinical, on='PatientID', how='outer')


def plot_km(dataframe, parameter, threshtype="median"):



    df1 = df[df["deadstatus.event"] == 1]
    df2 = df[df["deadstatus.event"] == 0]

    time = df["Survival.time"]
    censoring = df["deadstatus.event"]
    labx = df["Compactness2"] > df["Compactness2"].median()

    out = km.fit(time, censoring, labx)
    km.plot(out)
    plt.show()









