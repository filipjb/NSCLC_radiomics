import pydicom as dicom

dicom_path = "C:/Users/filip/Desktop/image-data/manifest-Lung1/NSCLC-Radiomics/LUNG1-001/" \
             "09-18-2008-StudyID-NA-69331/0.000000-NA-82046/1-044.dcm"

dcm = dicom.dcmread(dicom_path)

a = {3: [5, 333, 5], 5: [5, 5, 5], 2: "array2", 1: "array1", 4: "array4"}

sort = {k: v for k, v in sorted(a.items(), key=lambda item: item[0])}

print(list(sort.values()))
