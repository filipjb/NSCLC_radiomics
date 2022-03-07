import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


def plot_km(dataframe, parameter: str, groupname: str, xlim=1500):

    group1 = dataframe[dataframe[parameter] > dataframe[parameter].median()]
    group2 = dataframe[dataframe[parameter] <= dataframe[parameter].median()]

    t1 = group1["Survival.time"]
    t2 = group2["Survival.time"]
    e1 = group1["deadstatus.event"]
    e2 = group2["deadstatus.event"]

    kmf = KaplanMeierFitter()

    kmf.fit(t1, e1, label=f"{parameter} > median")

    ax = kmf.plot_survival_function(ci_show=False, color="blue")
    ax.set_xlim(0, xlim)

    kmf.fit(t2, e2, label=f"{parameter} <= median")
    kmf.plot_survival_function(ax=ax, ci_show=False, color="red")

    lr_result = logrank_test(t1, t2, e1, e2)
    pval = lr_result.p_value

    plt.title(f"{groupname} {parameter}, Logrank P-value = {pval.__round__(5)}")
    plt.ylabel("Survival probability")
    plt.xlabel("Survival time (days)")

    plt.show()


if __name__ == '__main__':

    lung1_shape = pd.read_csv(r"feature_files\lung1_shape.csv")
    lung1_firstorder = pd.read_csv(r"feature_files\lung1_firstorder.csv")
    lung1_glrlm = pd.read_csv(r"feature_files\lung1_GLRLM.csv")
    lung1_hlh = pd.read_csv(r"feature_files\lung1_HLH_GLRLM.csv")

    lung1_clinical = pd.read_csv(
        r"C:\Users\filip\Desktop\radiomics_data\NSCLC Radiomics Lung1.clinical-version3-Oct 2019.csv"
    )
    disq_patients = ["LUNG1-014", "LUNG1-021", "LUNG1-085", "LUNG1-095", "LUNG1-194", "LUNG1-128"]
    for i in disq_patients:  # Removing disqualified patients from clinical df
        lung1_clinical = lung1_clinical.drop(lung1_clinical[lung1_clinical.PatientID == i].index[0])

    lung1_firstorder = pd.merge(lung1_firstorder, lung1_clinical, on="PatientID", how="outer")
    lung1_shape = pd.merge(lung1_shape, lung1_clinical, on="PatientID", how="outer")
    lung1_glrlm = pd.merge(lung1_glrlm, lung1_clinical, on="PatientID", how="outer")
    lung1_hlh = pd.merge(lung1_hlh, lung1_clinical, on="PatientID", how="outer")


