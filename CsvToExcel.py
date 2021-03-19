import os
import glob
import csv
from xlsxwriter.workbook import Workbook
import pandas as pd

def combineallexcel(self):
    all_data = pd.DataFrame()
    for f in glob.glob("data/*.xlsx"):
        df = pd.read_excel(f)
        all_data = all_data.append(df.transpose(), ignore_index=True)
    all_data.to_csv("all_data.csv")

for csvfile in glob.glob(os.path.join('MissingBench/', '*.csv')):
    print (csvfile)
    filename ="MissingBench/"+ os.path.basename(csvfile)[0:-4] + '.xlsx'
    workbook = Workbook(filename)
    worksheet = workbook.add_worksheet()
    with open(csvfile, 'rt', encoding='utf8') as f:
        reader = csv.reader(f)
        for r, row in enumerate(reader):
            for c, col in enumerate(row):
                worksheet.write(r, c, col)
    workbook.close()