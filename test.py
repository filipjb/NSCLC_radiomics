import pydicom as dicom
import numpy as np
import matplotlib.pyplot as plt

#%%
path = r"C:\Users\filip\Desktop\test\RS.1.2.246.352.205.4628746736953205655.4330711959846355332.dcm"

ds = dicom.dcmread(path)

dic = dict()

for entry in ds.StructureSetROISequence:
    dic.update({entry.ROIName: int(entry.ROINumber)})

a = ds.ROIContourSequence

print(dic)
