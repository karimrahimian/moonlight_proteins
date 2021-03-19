from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
import numpy as np

import pandas as pd
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, confusion_matrix, \
    matthews_corrcoef, accuracy_score, roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split, KFold, train_test_split, train_test_split

from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, \
    GradientBoostingClassifier, VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import random
import glob

class MisClassify():
    def __init__(self):
        self.evaluationName = ['Precision', 'F1Score', 'Accuracy', 'Recall', 'Matt', 'Auc']
        self.run_times = 100
        self.threshould = 90

    def multiclass_roc_auc_score(self, y_test, y_pred, average="macro"):
        lb = LabelBinarizer()
        lb.fit(y_test)
        y_test = lb.transform(y_test)
        y_pred = lb.transform(y_pred)
        return roc_auc_score(y_test, y_pred, average=average)

    def misClassifyFrequency(self, ytest, ypred, idx):
        mis = []
        for i in range(len(ytest)):
            if (ytest[i] != ypred[i]):
                mis.append(idx[i])
        return mis

    def doClassify(self, X, y, classifier, nfold=10):
        Data = X
        misclassify = np.zeros(len(y))
        evlP = np.zeros((10,6))
        k = 0
        kf = KFold(n_splits=nfold, shuffle=True, random_state=random.randint(1, 100))
        for train_index, test_index in kf.split(Data):
            classifier.fit(Data[train_index], y[train_index])
            y_pred = classifier.predict(Data[test_index])
            y_test = y[test_index]

            evlP[k][0] = (precision_score(y_test, y_pred, average='micro'))
            evlP[k][1] = (f1_score(y_test, y_pred, average='macro'))
            evlP[k][2] = (accuracy_score(y_test, y_pred))
            evlP[k][3] = (recall_score(y_test, y_pred, average="weighted"))
            evlP[k][4] = (matthews_corrcoef(y_test, y_pred))
            evlP[k][5] = self.multiclass_roc_auc_score(y_test, y_pred)
            # evlP[k][5]= 0

            # cm = confusion_matrix(y_test, y_pred)
            k += 1
            mis = self.misClassifyFrequency(y_test, y_pred, test_index)
            for item in mis:
                misclassify[item] += 1

        average = evlP.mean(axis=0)
        average = np.squeeze(np.asarray(average))
        modelparams = pd.DataFrame({'Evaluating Function': self.evaluationName, 'Values': average})
        return modelparams, misclassify

    def applyModel(self, X, y, model):
        misclassify = []
        for i in range(len(y)):
            misclassify.append(0)

        for i in range(self.run_times):
            params, mis = self.doClassify(X, y, model)
            for i, item in enumerate(mis):
                if (item >= 1):
                    misclassify[i] += 1

        return misclassify

    def RemoveMisAndRunClassify(self, dataframe, model, miss ):
        ncol = dataframe.values.shape[1]
        rawdata = np.array(dataframe.to_numpy())
        y = rawdata[:, ncol - 1]
        y = y.astype(np.float)
        selIndex = []
        for i, item in enumerate(miss):
            if (item < self.threshould):
                selIndex.append(i)
        X = rawdata[selIndex, 1:ncol - 1]
        X = X.astype(np.float)
        y = rawdata[selIndex, ncol - 1]
        y = y.astype(np.float)
        X = StandardScaler().fit_transform(X)
        evalP = np.zeros((100,6))

        for i in range(100):
            r0 ,mismis = obj.doClassify(X, y, model)
            evalP[i,:] = r0.to_numpy()[:, 1]

        evalP = evalP.mean(axis=0)

        return evalP

    def RemoveCorrelated(self, dataframe):
        corr_matrix = dataframe.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = {}
        for i in range(upper.values.shape[0]):
            for j in range(i + 1, upper.values.shape[0]):
                if upper.values[i, j] >= 0.50:
                    to_drop[upper.columns[j]] = 1

        uncorrelated_data = dataframe.drop(to_drop.keys(), axis=1)
        return uncorrelated_data

    def FindIndex(self, missing, thresuld):
        indexes = []
        for i, item in enumerate(missing):
            if (item > thresuld):
                indexes.append(i)
        return indexes
    def MakeModel(self):
        ada = AdaBoostClassifier(n_estimators=100, base_estimator=None, learning_rate=1, random_state=1)
        knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
        nivebase = GaussianNB()
        dt = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
        lg = LogisticRegression(random_state=0)
        svclassifier = SVC(kernel='rbf')
        randomforest = RandomForestClassifier(n_estimators=10)  # Train the model on training data
        mlp = MLPClassifier(hidden_layer_sizes=(20, 3), max_iter=150, alpha=1e-4, solver='sgd', verbose=10, tol=1e-4,
                            random_state=1, learning_rate_init=.1)
        return ada,knn,nivebase,dt,lg,svclassifier,randomforest,mlp;
    def ReadExcelFile(self, filename, sheetname):
        dataframe = pd.read_excel(io=filename, sheet_name=sheetname)
        uncorrelated = self.RemoveCorrelated(dataframe)
        #       nrow = uncorrelated.values.shape[0]
        #       colheader = list(uncorrelated.columns.values)
        #       PID = rawdata[:, 0]

        ncol = uncorrelated.values.shape[1]
        rawdata = np.array(uncorrelated.to_numpy())
        X = rawdata[:, 1:ncol - 1]
        X = X.astype(np.float)
        y = rawdata[:, ncol - 1]
        y = y.astype(np.float)
        X = RobustScaler().fit_transform(X)
        return X, y
    def ApplyAllMissing(self, path):
        onlyfiles = glob.glob(path)
        totalAUC = pd.DataFrame(
            {'Features': [''], 'Ada': [0], 'Knn': [0], 'NB': [0], 'DT': [0], 'LR': [0], 'SVM': [0], 'RF': [0],
             'MLP': [0]})
        totalACC = pd.DataFrame(
            {'Features': [''], 'Ada': [0], 'Knn': [0], 'NB': [0], 'DT': [0], 'LR': [0], 'SVM': [0], 'RF': [0],
             'MLP': [0]})
        totalRecall = pd.DataFrame(
            {'Features': [''], 'Ada': [0], 'Knn': [0], 'NB': [0], 'DT': [0], 'LR': [0], 'SVM': [0], 'RF': [0],
             'MLP': [0]})
        totalPrecision = pd.DataFrame(
            {'Features': [''], 'Ada': [0], 'Knn': [0], 'NB': [0], 'DT': [0], 'LR': [0], 'SVM': [0], 'RF': [0],
             'MLP': [0]})
        totalf1 = pd.DataFrame(
            {'Features': [''], 'Ada': [0], 'Knn': [0], 'NB': [0], 'DT': [0], 'LR': [0], 'SVM': [0], 'RF': [0],
             'MLP': [0]})
        totalmatt = pd.DataFrame(
            {'Features': [''], 'Ada': [0], 'Knn': [0], 'NB': [0], 'DT': [0], 'LR': [0], 'SVM': [0], 'RF': [0],
             'MLP': [0]})
        for i, item in enumerate(onlyfiles):
            sheetname = os.path.basename(item).split('.')[0]
            print('processing {}'.format(sheetname))
            dataframe = pd.read_excel(io=item, sheet_name="Sheet1")
            uncorrelated = self.RemoveCorrelated(dataframe)
            nrow = uncorrelated.values.shape[0]
            ncol = uncorrelated.values.shape[1]
            colheader = list(uncorrelated.columns.values)
            rawdata = np.array(uncorrelated.to_numpy())
            PID = rawdata[:, 0]
            X = rawdata[:, 1:ncol - 1]
            X = X.astype(np.float)
            y = rawdata[:, ncol - 1]
            y = y.astype(np.float)
            X = StandardScaler().fit_transform(X)

            try:
                ada, knn, nivebase, dt, lg, svclassifier, randomforest, mlp = self.MakeModel()

                misada = obj.applyModel(X, y, ada)
                misknn = obj.applyModel(X, y, knn)
                misnb = obj.applyModel(X, y, nivebase)
                misdt = obj.applyModel(X, y, dt)
                mislg = obj.applyModel(X, y, lg)
                missvm = obj.applyModel(X, y, svclassifier)
                misrf = obj.applyModel(X, y, randomforest)
                mismlp = obj.applyModel(X, y, mlp)


                #r0 = obj.RemoveMisAndRunClassify(uncorrelated, ada, misada)
                r1 = obj.RemoveMisAndRunClassify(uncorrelated, knn, misknn)
                r2 = obj.RemoveMisAndRunClassify(uncorrelated, nivebase, misnb)
                r3 = obj.RemoveMisAndRunClassify(uncorrelated, dt, misdt)
                r4 = obj.RemoveMisAndRunClassify(uncorrelated, lg, mislg)
                r5 = obj.RemoveMisAndRunClassify(uncorrelated, svclassifier, missvm)
                r6 = obj.RemoveMisAndRunClassify(uncorrelated, randomforest, misrf)
                #r7 = obj.RemoveMisAndRunClassify(uncorrelated, mlp, mismlp)

                result = pd.DataFrame({'Evaluating Function': self.evaluationName,
                                       'Ada': r1,
                                       'KNN': r1,
                                       'NB': r2,
                                       'DT': r3,
                                       'LR': r4,
                                       'SVM': r5,
                                       'RF': r6,
                                       'MLP': r1})
                self.evaluationName = ['Precision', 'F1Score', 'Accuracy', 'Recall', 'Matt', 'Auc']

                totalPrecision.loc[i] = [sheetname] + list(result.values[0, 1:])
                totalf1.loc[i] = [sheetname] + list(result.values[1, 1:])
                totalACC.loc[i] = [sheetname] + list(result.values[2, 1:])
                totalRecall.loc[i] = [sheetname] + list(result.values[3, 1:])
                totalmatt.loc[i] = [sheetname] + list(result.values[4, 1:])
                totalAUC.loc[i] = [sheetname] + list(result.values[5, 1:])
            except:
                import sys
                print(sys.exc_info()[0])
                pass
        totalPrecision.to_csv("Result/totalPrecision.csv")
        totalf1.to_csv("Result/totalF1.csv")
        totalACC.to_csv("Result/totalACC.csv")
        totalRecall.to_csv("Result/totalRecall.csv")
        totalmatt.to_csv("Result/totalMatt.csv")
        totalAUC.to_csv("Result/totalAUC.csv")

cops_proteins = "TestData/*.xlsx"
obj = MisClassify()
obj.ApplyAllMissing(cops_proteins)


