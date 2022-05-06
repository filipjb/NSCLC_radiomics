import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import numpy as np
from lifelines import CoxPHFitter
import re
from resampy import resample
from scipy.stats import ks_2samp, cramervonmises_2samp, mannwhitneyu, pearsonr
from matplotlib.legend import Legend
from lifelines.utils import k_fold_cross_validation
from sklearn import tree, linear_model
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.feature_selection import SelectFromModel, SelectKBest
import pingouin as pg

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
# 14_radiomics_HUH is large volume outlier
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

    lin1, lin2 = plot_km(df, feature, df[feature].median(), "Lung1")

    lin3, = plt.plot(ref_over[0], ref_over[1], color="red", linestyle="--")
    lin4, = plt.plot(ref_under[0], ref_under[1], color="red")

    plt.gca().legend([lin2, lin4], ["Validation", "Aerts et al."], loc=1)

    leg = Legend(plt.gca(), [lin4, lin3], ["<= median", "> median"], loc=3)
    plt.gca().add_artist(leg)

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
    l1, = plt.plot(kmf.survival_function_.index, kmf.survival_function_["KM_estimate"], color="blue",
                   linestyle="--", label="> median")

    kmf.fit(t2, e2)
    l2, = plt.plot(kmf.survival_function_.index, kmf.survival_function_["KM_estimate"], color="blue",
                   label="<= median")

    # TODO if logrank test still unreliable, this test of our data can be used to quantify difference
    lr_result = logrank_test(t1, t2, e1, e2)
    pval = lr_result.p_value

    plt.gca().set_xlim(0, xlim)
    plt.gca().legend([l2, l1], ["<= median", "> median"])
    plt.legend()
    plt.title(f"{groupname} {parameter}, Logrank P-value = {pval.__round__(5)}")
    plt.ylabel("Survival probability")
    plt.xlabel("Survival time (days)")

    fig.set_figwidth(8)
    fig.set_figheight(5)

    return l1, l2,


def signature_cox_model(modeltype="radiomics", mute=False):
    # --------------- dfs for radiomics model ---------------- #
    energy = pd.DataFrame(lung1_firstorder["Energy"] > lung1_firstorder["Energy"].median())
    comp = lung1_shape["Compactness2"] > lung1_shape["Compactness2"].median()
    text = lung1_glrlm["GrayLevelNonUniformity"] > lung1_glrlm["GrayLevelNonUniformity"].median()
    wave = lung1_hlh["HLH GrayLevelNonUniformity"] > lung1_hlh["HLH GrayLevelNonUniformity"].median()
    time = lung1_firstorder["Survival.time"]
    event = lung1_firstorder["deadstatus.event"]

    # ---------------- dfs for basic clinical moodel ------------- #
    age = pd.DataFrame(lung1_firstorder["age"].round())
    sex = pd.get_dummies(lung1_firstorder["gender"]).drop("male", axis=1).rename(columns={"female": "gender"})

    stage = list()
    for row in lung1_firstorder["Overall.Stage"]:
        if row == "I":
            stage.append({"Overall.stage": 1})
        elif row == "II":
            stage.append({"Overall.stage": 2})
        elif row == "IIIa" or row == "IIIb":
            stage.append({"Overall.stage": 3})
        else:
            stage.append({"Overall.stage": pd.NA})
    stage = pd.DataFrame(stage)

    # -------------- dfs for tnm model --------------- #
    t = pd.DataFrame(lung1_firstorder["clinical.T.Stage"]).astype(int)
    n = lung1_firstorder["Clinical.N.Stage"]
    m = lung1_firstorder["Clinical.M.Stage"]

    volume = pd.DataFrame(lung1_shape["VoxelVolume"])

    if modeltype == "radiomics":
        df = energy.join([comp, text, wave, time, event])
        # Renaming for brevity
        df.rename(columns={"GrayLevelNonUniformity": "GLNU", "HLH GrayLevelNonUniformity": "HLH GLNU"}, inplace=True)

    elif modeltype == "clinical":
        df = age.join([sex, stage, time, event])
        df = df.dropna()

    elif modeltype == "tnm":
        print(t)
        print(n)
        print(m)
        df = t.join([n, m, stage, time, event])
        df = df.dropna()

    elif modeltype == "volume":
        df = volume.join([time, event])

    else:
        print("No valid modeltype")
        quit()

    fitter = CoxPHFitter()
    fitter.fit(df, duration_col="Survival.time", event_col="deadstatus.event")

    if not mute:
        fitter.print_summary(decimals=3)
        fitter.plot()  # Plots regression coefficients with 95% confidence intervals
        plt.show()

    return fitter, df


def plot_signature_km():
    df = pd.DataFrame(lung1_firstorder["Energy"])
    df = df.join(
        [lung1_shape["Compactness2"], lung1_glrlm["GrayLevelNonUniformity"], lung1_hlh["HLH GrayLevelNonUniformity"]]
    )

    huh_df = pd.concat(
        [huh_firstorder["Energy"], huh_shape["Compactness2"], huh_glrlm["GrayLevelNonUniformity"],
         huh_hlh["HLH GrayLevelNonUniformity"]], axis=1)

    cph, train = signature_cox_model(modeltype="radiomics", mute=True)
    weights = cph.params_

    combined = pd.DataFrame(df["Energy"] * weights["Energy"] + df["Compactness2"] * weights["Compactness2"]
                            + df["GrayLevelNonUniformity"] * weights["GLNU"]
                            + df["HLH GrayLevelNonUniformity"] * weights["HLH GLNU"])

    huh_combined = pd.DataFrame(huh_df["Energy"] * weights["Energy"] + huh_df["Compactness2"] * weights["Compactness2"]
                            + huh_df["GrayLevelNonUniformity"] * weights["GLNU"]
                            + huh_df["HLH GrayLevelNonUniformity"] * weights["HLH GLNU"])

    combined = combined.join([lung1_firstorder["Survival.time"], lung1_firstorder["deadstatus.event"]])
    # Reuturning this to allow combined signature to be used in other functions
    total_combined = combined

    threshold = combined[0].median()
    group1 = combined[combined[0] > threshold]
    group2 = combined[combined[0] <= threshold]

    t1 = group1["Survival.time"]
    t2 = group2["Survival.time"]
    e1 = group1["deadstatus.event"]
    e2 = group2["deadstatus.event"]

    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots()

    kmf.fit(t1, e1)
    lin1, = plt.plot(kmf.survival_function_.index, kmf.survival_function_["KM_estimate"],
                     color="blue", linestyle="--", label="> median")

    kmf.fit(t2, e2)
    lin2, = plt.plot(kmf.survival_function_.index, kmf.survival_function_["KM_estimate"],
                     color="blue", label="<= median")

    ref_over = pd.read_csv("automeris_coords/lung1overall_overmedian.csv", delimiter=";", decimal=",", header=None)
    ref_under = pd.read_csv("automeris_coords/lung1overall_undermedian.csv", delimiter=";", decimal=",", header=None)
    lin3, = plt.plot(ref_over[0], ref_over[1], color="red", linestyle="--")
    lin4, = plt.plot(ref_under[0], ref_under[1], color="red")

    plt.gca().set_xlim(0, 1500)
    plt.gca().legend([lin2, lin1], ["<= median", "> median"])
    plt.legend()
    plt.title(f"Combined signature")
    plt.ylabel("Survival probability")
    plt.xlabel("Survival time (days)")

    plt.gca().legend([lin2, lin4], ["Validation", "Aerts et al."], loc=1)
    leg = Legend(plt.gca(), [lin4, lin3], ["<= median", "> median"], loc=3)
    plt.gca().add_artist(leg)
    lines = leg.get_lines()
    for line in lines:
        line.set_color("black")

    fig.set_figwidth(8)
    fig.set_figheight(5)

    return total_combined, huh_combined


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


def test_featuregroup(df1, df2, k=None, log=False, tight=False):
    df1 = df1.drop(["PatientID", "age", "Overall.Stage", "Histology", "gender", "deadstatus.event",
                    "Survival.time", "Unnamed: 0", "Clinical.M.Stage", "clinical.T.Stage", "Clinical.N.Stage"], axis=1)
    df2 = df2.drop(["PatientID", "age", "Overall.Stage", "Histology", "gender", "deadstatus.event",
                    "Survival.time", "Unnamed: 0", "Clinical.M.Stage", "clinical.T.Stage", "Clinical.N.Stage"], axis=1)

    result = dict()
    for col in df1:
        col1 = df1[col]
        col2 = df2[col]
        stat, p = ks_2samp(col1, col2)
        result.update({col: p})
    result = pd.DataFrame(result, index=[0]).transpose()

    if k is not None:
        result = result.nlargest(k, columns=0)
    result = result.sort_values(by=0)

    fig, ax = plt.subplots()
    if log:
        plt.xscale("log")

    cc = list(map(lambda x: 'indianred' if x < 0.05 else 'olivedrab', result[0]))
    bars = ax.barh(result.index, result[0], edgecolor="black", color=cc)

    plt.xlabel("p-value")
    ax.bar_label(bars)
    plt.axvline(x=0.05, linewidth=1.7, color="black", linestyle="--")

    if tight:
        plt.tight_layout()
    plt.show()
    return result


def test_all_features(k=10, lung1_la=False):
    if lung1_la:
        lung1_df = la_lung1_firstorder.merge(la_lung1_shape)
        lung1_df = lung1_df.merge(la_lung1_glrlm)
        lung1_df = lung1_df.merge(la_lung1_hlh)

    else:
        lung1_df = lung1_firstorder.merge(lung1_shape)
        lung1_df = lung1_df.merge(lung1_glrlm)
        lung1_df = lung1_df.merge(lung1_hlh)

    lung1_df = lung1_df.drop(["PatientID", "age", "Overall.Stage", "Histology", "gender", "deadstatus.event",
                              "Survival.time", "Unnamed: 0", "Clinical.M.Stage", "clinical.T.Stage",
                              "Clinical.N.Stage"], axis=1)
    huh_df = huh_firstorder.merge(huh_shape)
    huh_df = huh_df.merge(huh_glrlm)
    huh_df = huh_df.merge(huh_hlh)

    huh_df = huh_df.drop(["PatientID", "age", "Overall.Stage", "Histology", "gender", "deadstatus.event",
                          "Survival.time", "Unnamed: 0", "Clinical.M.Stage", "clinical.T.Stage", "Clinical.N.Stage"],
                         axis=1)

    result = dict()
    for col in lung1_df:
        col1 = lung1_df[col]
        col2 = huh_df[col]
        stat, p = ks_2samp(col1, col2)
        result.update({col: p})

    result = pd.DataFrame(result, index=[0]).transpose()
    result = result.nlargest(k, columns=0)
    result = result.sort_values(by=0)

    cc = list(map(lambda x: 'indianred' if x < 0.05 else 'olivedrab', result[0]))

    fig, ax = plt.subplots()
    bars = ax.barh(result.index, result[0].round(3), height=0.5, edgecolor="black", color=cc)
    plt.xlabel("p-value")
    ax.bar_label(bars)
    plt.axvline(x=0.05, linewidth=1.7, color="black", linestyle="--")
    if lung1_la:
        plt.title(f"{k} most similarly distributed features, LA")
    else:
        plt.title(f"{k} most similarly distributed features")
    plt.tight_layout()
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


def regressor_selection(df_list: list, regtype="tree"):
    lst = list()
    for featuregroup in df_list:
        x = featuregroup.drop(
            labels=["Unnamed: 0", "Survival.time", "PatientID", "Overall.Stage", "Histology", "deadstatus.event", "gender",
                    "clinical.T.Stage", "Clinical.N.Stage", "Clinical.M.Stage", "age"],
            axis=1
        )
        lst.append(x)
    X = pd.concat(lst, axis=1)

    # Removing duplicate columns
    X = X.loc[:, ~X.columns.duplicated()]
    Y = df_list[0]["Survival.time"]
    # Data split
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, random_state=666, train_size=0.8)

    if regtype == "lasso":
        model = linear_model.Lasso(max_iter=1000, random_state=0)
        model.fit(X_train, Y_train)

    elif regtype == "tree":
        model = tree.ExtraTreeRegressor(random_state=0)
        model.fit(X_train, Y_train)

    else:
        print("Error: Invalid modeltype")
        quit()

    print(f"Model score for {model.__str__()}: {model.score(X_val, Y_val)}")

    selector = SelectKBest(k=15)
    selector.fit(X, Y)
    print(selector.get_feature_names_out())

    Y_pred = model.predict(X_val)
    print(tree)
    plt.scatter(X_val.index, Y_val)
    plt.scatter(X_val.index, Y_pred)
    plt.show()

    imps = pd.DataFrame({"Feature": list(X.columns), "Importance": model.feature_importances_})
    plt.barh([x for x in list(X.columns)], imps["Importance"])
    plt.show()


def separate_LA_lung1():
    new_firstorder = lung1_firstorder[
        (lung1_firstorder["Overall.Stage"] != "I") & (lung1_firstorder["Overall.Stage"] != "II")
        ]

    new_shape = lung1_shape[
        (lung1_shape["Overall.Stage"] != "I") & (lung1_shape["Overall.Stage"] != "II")
        ]

    new_glrlm = lung1_glrlm[
        (lung1_glrlm["Overall.Stage"] != "I") & (lung1_glrlm["Overall.Stage"] != "II")
        ]

    new_hlh = lung1_hlh[
        (lung1_hlh["Overall.Stage"] != "I") & (lung1_hlh["Overall.Stage"] != "II")
        ]

    return new_firstorder, new_shape, new_glrlm, new_hlh


def featurevolume_correlation(featurname, log=False):
    lung1_volume = lung1_shape["VoxelVolume"]
    huh_volume = huh_shape["VoxelVolume"]

    try:
        lung1_feat = lung1_firstorder[featurname]
        huh_feat = huh_firstorder[featurname]
    except KeyError:
        pass
    try:
        lung1_feat = lung1_shape[featurname]
        huh_feat = huh_shape[featurname]
    except KeyError:
        pass
    try:
        lung1_feat = lung1_glrlm[featurname]
        huh_feat = huh_glrlm[featurname]
    except KeyError:
        pass
    try:
        lung1_feat = lung1_hlh[featurname]
        huh_feat = huh_hlh[featurname]
    except KeyError:
        pass

    lung1corr = pearsonr(lung1_feat, lung1_volume)

    fig, ax = plt.subplots()

    if log:
        plt.xscale("log")
        plt.yscale("log")

    ax.scatter(lung1_volume, lung1_feat, edgecolors="black", s=70, label="Lung1")
    ax.scatter(huh_volume, huh_feat, edgecolors="black", s=70, color="orange", label="HUH")
    plt.legend()
    plt.xlabel("Volume")
    plt.ylabel(featurname)
    fig.set_figwidth(10)
    fig.set_figheight(6)
    plt.title(f"Lung1 Pearson correlation coefficient = {lung1corr[0]}, p = {lung1corr[1].round(4)}\n")


def signaturevolume_correlation(log=False):

    lung1_sig, huh_sig = plot_signature_km()
    lung1_sig = lung1_sig[0]
    huh_sig = huh_sig[0]

    lung1_volume = lung1_shape["VoxelVolume"]
    huh_volume = huh_shape["VoxelVolume"]

    lung1corr = pearsonr(lung1_sig, lung1_volume)

    fig, ax = plt.subplots()
    if log:
        plt.xscale("log")
        plt.yscale("log")

    ax.scatter(lung1_volume, lung1_sig, edgecolors="black", s=70, label="Lung1")
    ax.scatter(huh_volume, huh_sig, edgecolors="black", s=70, color="orange", label="HUH")
    plt.xlabel("Volume")
    plt.ylabel("Combined signature")
    fig.set_figwidth(10)
    fig.set_figheight(6)
    plt.title(f"Lung1 Pearson correlation coefficient = {lung1corr[0]}, p = {lung1corr[1].round(4)}\n")



la_lung1_firstorder, la_lung1_shape, la_lung1_glrlm, la_lung1_hlh = separate_LA_lung1()


if __name__ == '__main__':
    plt.style.use("bmh")
    # cph, train = signature_cox_model(modeltype="volume", mute=False)
    #tree = regressor_selection([lung1_firstorder, lung1_shape, lung1_glrlm, lung1_hlh], regtype="tree")

    #test_featuregroup(lung1_hlh, huh_hlh, log=True, tight=True)

    signaturevolume_correlation(log=True)
    plt.show()
