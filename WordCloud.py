import glob
import pandas as pd
import numpy as np
from collections import Counter

path = "Result/TotalMissing.xlsx"
onlyfiles = glob.glob(path)

totalproteins = list()

for i, item in enumerate(onlyfiles):
    sheetname = "Sheet1"
    #print('processing {}'.format(item))
    dataframe = pd.read_excel(io=item, sheet_name=sheetname)
    rawdata = np.array(dataframe.to_numpy())
    proteins = rawdata[:,1]
    for p in proteins:
        t= str(p).split("|")
        print(t[1]+" "+ t[2])
        if (len(t)==3):
            totalproteins.append(t[1].strip().upper())
            if (t[1]=="P69786"):
                print(item)
        else:
            totalproteins.append(t[0].strip().lower())
            if (t[0]=="P69786"):
                print(item)
print (Counter(totalproteins))