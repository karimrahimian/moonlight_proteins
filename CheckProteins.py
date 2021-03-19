import glob
import pandas as pd
import numpy as np
from collections import Counter


dataframe = pd.read_excel(io="BenckMarkData/All/xlsx/ctd.xlsx", sheet_name="Sheet1")
rawdata = np.array(dataframe.to_numpy())
proteinID = set(rawdata[: ,0])

dataframe = pd.read_excel(io="Data/All/xlsx/IF.xlsx", sheet_name="Sheet1")
rawdata = np.array(dataframe.to_numpy())
proteinID1 = set(rawdata[: ,0])

tt= proteinID.intersection(proteinID1)
print (len(tt))
print (tt)