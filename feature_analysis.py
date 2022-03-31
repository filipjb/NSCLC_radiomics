import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import numpy as np
from lifelines import CoxPHFitter
import re
from scipy.stats import ks_2samp

lung1_firstorder = pd.read_csv(r"feature_files\lung1_firstorder.csv")
lung1_shape = pd.read_csv(r"feature_files\lung1_shape.csv")
lung1_glrlm = pd.read_csv(r"feature_files\lung1_GLRLM.csv")
lung1_hlh = pd.read_csv(r"feature_files\lung1_HLH_GLRLM.csv")

huh_firstorder = pd.read_csv(r"feature_files\HUH_firstorder.csv")
huh_shape = pd.read_csv(r"feature_files\HUH_shape.csv")
huh_glrlm = pd.read_csv(r"feature_files\HUH_GLRLM.csv")
huh_hlh = pd.read_csv(r"feature_files\HUH_HLH_GLRLM.csv")

huh_clinical = pd.read_csv(
    r"HUH_clinical.csv", delimiter=";"
)

lung1_clinical = pd.read_csv(
    r"NSCLC Radiomics Lung1.clinical-version3-Oct 2019.csv"
)
disq_lung1 = ["LUNG1-014", "LUNG1-021", "LUNG1-085", "LUNG1-095", "LUNG1-194", "LUNG1-128"]
for i in disq_lung1:
    lung1_clinical = lung1_clinical.drop(lung1_clinical[lung1_clinical.PatientID == i].index[0])

disq_huh = ["26_radiomics_HUH", "27_radiomics_HUH", "28_radiomics_HUH"]
for i in disq_huh:
    huh_clinical = huh_clinical.drop(huh_clinical[huh_clinical.PatientID == i].index[0])

lung1_firstorder = pd.merge(lung1_firstorder, lung1_clinical, on="PatientID", how="outer")
lung1_shape = pd.merge(lung1_shape, lung1_clinical, on="PatientID", how="outer")
lung1_glrlm = pd.merge(lung1_glrlm, lung1_clinical, on="PatientID", how="outer")
lung1_hlh = pd.merge(lung1_hlh, lung1_clinical, on="PatientID", how="outer")

huh_firstorder = pd.merge(huh_firstorder, huh_clinical, on="PatientID", how="inner")
huh_shape = pd.merge(huh_shape, huh_clinical, on="PatientID", how="inner")
huh_glrlm = pd.merge(huh_glrlm, huh_clinical, on="PatientID", how="inner")
huh_hlh = pd.merge(huh_hlh, huh_clinical, on="PatientID", how="inner")


def plot_km(dataframe, parameter: str, threshold, groupname: str, xlim=1500):

    group1 = dataframe[dataframe[parameter] > threshold]
    group2 = dataframe[dataframe[parameter] <= threshold]

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


def signature_cox_model():
    energy = pd.DataFrame(lung1_firstorder["Energy"] > lung1_firstorder["Energy"].median())
    comp = lung1_shape["Compactness2"] > lung1_shape["Compactness2"].median()
    text = lung1_glrlm["GrayLevelNonUniformity"] > lung1_glrlm["GrayLevelNonUniformity"].median()
    wave = lung1_hlh["HLH GrayLevelNonUniformity"] > lung1_hlh["HLH GrayLevelNonUniformity"].median()
    time = lung1_firstorder["Survival.time"]
    event = lung1_firstorder["deadstatus.event"]

    df = energy.join([comp, text, wave, time, event])

    fitter = CoxPHFitter()
    fitter.fit(df, duration_col="Survival.time", event_col="deadstatus.event")
    fitter.print_summary()
    fitter.plot()
    plt.show()



def plot_signature_km(firstorder, shape, texture, wavelet):
    pass


# Comparing the histogram of a feature value across the two cohorts
# TODO hopefull perform a test here to quantify difference between histograms
def compare_histograms(df1, df2, featurename):
    if re.match("LUNG1", df1.iloc[0].PatientID):
        label1 = "Lung1"
        label2 = "HUH"
    else:
        label1 = "HUH"
        label2 = "Lung1"

    df1 = df1[featurename]
    df2 = df2[featurename]
    minimum = min(df1.min(), df2.min())
    maximum = max(df1.max(), df2.max())
    # Kolmogorov-Smirnov test
    stat, p = ks_2samp(df1, df2)

    binning = np.linspace(minimum, maximum, 30)
    fig, ax = plt.subplots()
    ax.hist(df1, density=True, alpha=0.7, bins=binning, edgecolor="black", label=label1)
    ax.hist(df2, density=True, alpha=0.7, bins=binning, edgecolor="black", label=label2)
    fig.set_figwidth(8)
    fig.set_figheight(5)
    plt.title(f"{featurename}, KS-statistic = {stat.__round__(5)}, p-value = {p.__round__(5)}")

    plt.legend()
    plt.show()


def test_all_features(df1, df2):
    df1 = df1.drop(["PatientID", "age", "Overall.Stage", "Histology", "gender", "deadstatus.event",
                    "Survival.time", "Unnamed: 0", "Clinical.M.Stage"], axis=1)
    df2 = df2.drop(["PatientID", "age", "Overall.Stage", "Histology", "gender", "deadstatus.event",
                    "Survival.time", "Unnamed: 0", "Clinical.M.Stage"], axis=1)

    result = dict()
    for col in df1:
        col1 = df1[col]
        col2 = df2[col]
        stat, p = ks_2samp(col1, col2)
        result.update({col: p})
    result = dict(sorted(result.items(), key=lambda x:x[1]))

    fig, ax = plt.subplots()
    bars = ax.barh(list(result.keys()), list(result.values()), edgecolor="black")
    plt.xlabel("p-value")
    ax.bar_label(bars)
    plt.show()
    return result


def thresholded_histograms(df, feature: str, clinical: str):
    thresh = df[feature].median()
    df1 = df[df[feature] <= thresh]
    df2 = df[df[feature] > thresh]

    df1 = df1[clinical]
    df2 = df2[clinical]

    n = df.nunique(axis=0)[clinical]
    binning = np.arange(-0.5, n+1, 1)
    lab = [f"{feature} <= median", f"{feature} > median"]

    plt.hist([df1, df2], bins=binning, color=["b", "r"], rwidth=0.5, label=lab)
    plt.title(f"{clinical} above and below the median value of {feature}")
    plt.xlabel(clinical)
    plt.ylabel("n")
    plt.legend()
    plt.show()


if __name__ == '__main__':

