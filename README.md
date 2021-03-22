# moonlight_proteins
mpCool is a set of machine learning tool to detect moonlighting proteins by using various of feature vectors. It is also containing a tool for protein outlier’s finder and protein outlier’s removal. 
How to Use
DoClassify
This file train some machine learning model (Ada, Knn, Nb, Mlp, Svm, DT, RF, LR) with various (38 different feature vector) extracted features from moonlighting and non-moonlighting proteins.
First save all the files you want in the data/xlsx folder in the current folder. Each file is an excel file format contains extracted features for all of protein (moonlight and non-MP). Each row in the excel files represent a protein and each column represent a feature. First and last column indicate protein id and label class respectively. 
Finally to run, it is enough to call the DoClassifyAllFeature function after creating an object of Moonlihg_Classifier class. In the following example is showing vividly:
moonlight_object = Moonlihg_Classifier()
moonlight_object.DoClassifyAllFeature(feature_xlsx_path)
the feature_xlsx_path is the path of excel files( for example data/xlsx). 

Note: the result will be saved in the result folder, so you need to make a folder with result  name’s in the current directory or in your working directory. 
DoClassify_100_runs:
Use this file to obtain 100*10 cross fold validation results. This file running  is very similar to DoClassify, after copying your feature files in the data/xlsx directory all of things you need to do are to call ApplyAllData(path). Path argument in this function refer to data/xslsx directory. The following example illustrate well:
path = "Data/All/xlsx/*.xlsx" 
moonlight_object = Moonlihg_Classifier()
moonlight_object.ApplyAllData(path)

 The result will be saved in result directory, so you need to build it in the current directory or in the working directory. 
RemoveOutliers:
To find and remove outlier and re run 10 fold cross validation use this file. The all pre configuration you need is to set the feature vector xlsx files path. Similar to other method the result will be saved in the result directory. 
OutlierFinders:
By using this file you could find outliers proteins a cross all machine learning model and feature vectors.
In this file two function are there:
FindCOP function find outliers just by Knn, Svm model with SAAC feature vector and Nb model with QSorder feature vector. Before calling this function be sure that SAAC.xlsx, QSrder.xslx are saved in 
FindAllCOP  function  find outliers among all model and all provided features in the COPData folder.
