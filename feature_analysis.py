import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import numpy as np
from lifelines import CoxPHFitter
import re
from scipy.stats import ks_2samp
from matplotlib.legend import Legend

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

disq_huh = ["26_radiomics_HUH", "27_radiomics_HUH", "28_radiomics_HUH", "14_radiomics_HUH"]
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


def compare_km(feature: str):
    if feature == "Energy":
        ref_over = pd.read_csv("automeris_coords/lung1energy_overmedian.csv", delimiter=";", decimal=",",
                               header=None)
        ref_under = pd.read_csv("automeris_coords/lung1energy_undermedian.csv", delimiter=";", decimal=",",
                                header=None)
        df = lung1_firstorder

    if feature == "Compactness2":
        ref_over = pd.read_csv("automeris_coords/lung1compactness_overmedian.csv", delimiter=";", decimal=",",
                               header=None)
        ref_under = pd.read_csv("automeris_coords/lung1compactness_undermedian.csv", delimiter=";", decimal=",",
                                header=None)
        df = lung1_shape

    if feature == "GrayLevelNonUniformity":
        ref_over = pd.read_csv("automeris_coords/lung1glnu_overmedian.csv", delimiter=";", decimal=",",
                               header=None)
        ref_under = pd.read_csv("automeris_coords/lung1glnu_undermedian.csv", delimiter=";", decimal=",",
                                header=None)
        df = lung1_glrlm

    if feature == "HLH GrayLevelNonUniformity":
        ref_over = pd.read_csv("automeris_coords/lung1hlhglnu_overmedian.csv", delimiter=";", decimal=",",
                               header=None)
        ref_under = pd.read_csv("automeris_coords/lung1hlhglnu_undermedian.csv", delimiter=";", decimal=",",
                                header=None)
        df = lung1_hlh

    lin1, lin2, xover, xunder = plot_km(df, feature, df[feature].median(), "Lung1")

    lin3, = plt.plot(ref_over[0], ref_over[1], color="red", linestyle="--")
    lin4, = plt.plot(ref_under[0], ref_under[1], color="red")

    plt.gca().legend([lin2, lin4], ["Validation", "Aerts et al."], loc=1)

    leg = Legend(plt.gca(), [lin4, lin3], ["<= median", "> median"], loc=3)
    plt.gca().add_artist(leg)

    ks13 = ks_2samp(lin1.get_xdata(), lin3.get_xdata())
    print(ks13)

    plt.title(f"{feature}")
    lines = leg.get_lines()
    for line in lines:
        line.set_color("black")

    plt.show()


def plot_km(dataframe, parameter: str, threshold, groupname: str, xlim=1500):
    group1 = dataframe[dataframe[parameter] > threshold]
    group2 = dataframe[dataframe[parameter] <= threshold]

    t1 = group1["Survival.time"]
    t2 = group2["Survival.time"]
    e1 = group1["deadstatus.event"]
    e2 = group2["deadstatus.event"]

    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots()

    kmf.fit(t1, e1)
    xover = kmf.survival_function_.index
    l1, = plt.plot(kmf.survival_function_.index, kmf.survival_function_["KM_estimate"], color="blue",
                   linestyle="--", label="> median")

    kmf.fit(t2, e2)
    xunder = kmf.survival_function_.index
    l2, = plt.plot(kmf.survival_function_.index, kmf.survival_function_["KM_estimate"], color="blue",
                   label="<= median")

    # TODO if logrank test still unreliable, this test of our data can be used to quantify difference
    lr_result = logrank_test(t1, t2, e1, e2)
    pval = lr_result.p_value

    plt.gca().set_xlim(0, xlim)
    # plt.gca().legend([l2, l1], ["<= median", "> median"])
    plt.legend()
    plt.title(f"{groupname} {parameter}, Logrank P-value = {pval.__round__(5)}")
    plt.ylabel("Survival probability")
    plt.xlabel("Survival time (days)")

    fig.set_figwidth(8)
    fig.set_figheight(5)

    return l1, l2, t1, t2


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
    result = dict(sorted(result.items(), key=lambda x: x[1]))

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
    binning = np.arange(-0.5, n + 1, 1)
    lab = [f"{feature} <= median", f"{feature} > median"]

    plt.hist([df1, df2], bins=binning, color=["b", "r"], rwidth=0.5, label=lab)
    plt.title(f"{clinical} above and below the median value of {feature}")
    plt.xlabel(clinical)
    plt.ylabel("n")
    plt.legend()
    plt.show()


if __name__ == '__main__':

    compare_km("Energy")

    #plot_km(lung1_firstorder, "Energy", lung1_firstorder["Energy"].median(), "Lung1")

    '''
    compare_histograms(lung1_firstorder, huh_firstorder, "Energy")
    compare_histograms(lung1_shape, huh_shape, "Compactness2")
    compare_histograms(lung1_glrlm, huh_glrlm, "GrayLevelNonUniformity")
    compare_histograms(lung1_hlh, huh_hlh, "HLH GrayLevelNonUniformity")
    '''

    #test_all_features(lung1_shape, huh_shape)
    #plt.show()



