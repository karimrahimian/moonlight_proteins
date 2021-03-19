from sklearn.preprocessing import RobustScaler
import numpy as np
import pandas as pd
import os
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, accuracy_score, roc_auc_score
from sklearn.model_selection import  KFold, train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import glob

class Moonlihg_Classifier():
    def __init__(self):
        self.evaluationName = ['Precision', 'F1Score', 'Accuracy', 'Recall', 'Matt', 'Auc']
    def multiclass_roc_auc_score(self, y_test, y_pred, average="macro"):
        lb = LabelBinarizer()
        lb.fit(y_test)
        y_test = lb.transform(y_test)
        y_pred = lb.transform(y_pred)
        return roc_auc_score(y_test, y_pred, average=average)
    def doClassifyTrainAndTest(self, xTrain,yTrain,xTest,yTest, classifier):
        evlP = np.zeros(6)
        classifier.fit(xTrain, yTrain)
        y_pred = classifier.predict(xTest)
        y_test =yTest

        evlP[0] = (precision_score(y_test, y_pred, average='micro'))
        evlP[1] = (f1_score(y_test, y_pred, average='macro'))
        evlP[2] = (accuracy_score(y_test, y_pred))
        evlP[3] = (recall_score(y_test, y_pred, average="weighted"))
        evlP[4] = (matthews_corrcoef(y_test, y_pred))
        evlP[5] = self.multiclass_roc_auc_score(y_test, y_pred)

        modelparams = pd.DataFrame({'Evaluating Function': self.evaluationName, 'Values': evlP})
        return modelparams
    def applyAllmodelTrainAndTest(self, xTrain,yTrain,xTest,yTest):
        ada = AdaBoostClassifier(n_estimators=200, base_estimator=None, learning_rate=0.1, random_state=1)
        knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
        nivebase = GaussianNB()
        dt = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
        lr = LogisticRegression(random_state=0)
        svclassifier = SVC(kernel='rbf', degree=3, C=2)
        randomforest = RandomForestClassifier(n_estimators=200)  # Train the model on training data
        mlp = MLPClassifier(hidden_layer_sizes=(20, 10, 5), max_iter=1000, alpha=1e-4, solver='sgd', verbose=0,
                            tol=1e-4, random_state=1, learning_rate_init=.08)

        paramada = self.doClassifyTrainAndTest(xTrain,yTrain,xTest,yTest, ada)
        paramknn = self.doClassifyTrainAndTest(xTrain,yTrain,xTest,yTest, knn)
        paramnb = self.doClassifyTrainAndTest(xTrain,yTrain,xTest,yTest, nivebase)
        paramdt = self.doClassifyTrainAndTest(xTrain,yTrain,xTest,yTest, dt)
        paramlr = self.doClassifyTrainAndTest(xTrain,yTrain,xTest,yTest, lr)
        paramsvm = self.doClassifyTrainAndTest(xTrain,yTrain,xTest,yTest, svclassifier)
        paramrf = self.doClassifyTrainAndTest(xTrain,yTrain,xTest,yTest, randomforest)
        parammlp = self.doClassifyTrainAndTest(xTrain,yTrain,xTest,yTest, mlp)

        result = pd.DataFrame({'Evaluating Function': self.evaluationName,
                               'Ada': paramada.to_numpy()[:, 1],
                               'KNN': paramknn.to_numpy()[:, 1],
                               'NB': paramnb.to_numpy()[:, 1],
                               'DT': paramdt.to_numpy()[:, 1],
                               'LR': paramlr.to_numpy()[:, 1],
                               'SVM': paramsvm.to_numpy()[:, 1],
                               'RF': paramrf.to_numpy()[:, 1],
                               'MLP': parammlp.to_numpy()[:, 1]})
        return result
    def doClassifyCrossValidation(self, X, y, classifier, nfold=5):
        Data = X
        evlP = [[0 for x in range(6)] for YY in range(nfold)]
        k = 0
        kf = KFold(n_splits=nfold, shuffle=False, random_state=1)
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
            k += 1

        average = np.matrix(evlP)
        average = average.mean(axis=0)
        average = np.squeeze(np.asarray(average))
        modelparams = pd.DataFrame({'Evaluating Function': self.evaluationName, 'Values': average})
        return modelparams
    def applyAllmodelCrossValidation(self, X, Y):
        ada = AdaBoostClassifier(n_estimators=200, base_estimator=None, learning_rate=0.1, random_state=1)
        knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
        nivebase = GaussianNB()
        dt = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
        lr = LogisticRegression(random_state=0)
        svclassifier = SVC(kernel='rbf', degree=3, C=2)
        randomforest = RandomForestClassifier(n_estimators=200)  # Train the model on training data
        mlp = MLPClassifier(hidden_layer_sizes=(20, 10, 5), max_iter=1000, alpha=1e-4, solver='sgd', verbose=0,
                            tol=1e-4, random_state=1, learning_rate_init=.08)

        paramada = self.doClassifyCrossValidation(X, Y, ada)
        paramknn = self.doClassifyCrossValidation(X, Y, knn)
        paramnb = self.doClassifyCrossValidation(X, Y, nivebase)
        paramdt = self.doClassifyCrossValidation(X, Y, dt)
        paramlr = self.doClassifyCrossValidation(X, Y, lr)
        paramsvm = self.doClassifyCrossValidation(X, Y, svclassifier)
        paramrf = self.doClassifyCrossValidation(X, Y, randomforest)
        parammlp = self.doClassifyCrossValidation(X, Y, mlp)

        result = pd.DataFrame({'Evaluating Function': self.evaluationName,
                               'Ada': paramada.to_numpy()[:, 1],
                               'KNN': paramknn.to_numpy()[:, 1],
                               'NB': paramnb.to_numpy()[:, 1],
                               'DT': paramdt.to_numpy()[:, 1],
                               'LR': paramlr.to_numpy()[:, 1],
                               'SVM': paramsvm.to_numpy()[:, 1],
                               'RF': paramrf.to_numpy()[:, 1],
                               'MLP': parammlp.to_numpy()[:, 1]})
        return result
    def applyAllmodelCrossValidation(self, X, Y):
        ada = AdaBoostClassifier(n_estimators=200, base_estimator=None, learning_rate=0.1, random_state=1)
        knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
        nivebase = GaussianNB()
        dt = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
        lr = LogisticRegression(random_state=0)
        svclassifier = SVC(kernel='rbf', degree=3, C=2)
        randomforest = RandomForestClassifier(n_estimators=200)  # Train the model on training data
        mlp = MLPClassifier(hidden_layer_sizes=(20, 10, 5), max_iter=1000, alpha=1e-4, solver='sgd', verbose=0,
                            tol=1e-4, random_state=1, learning_rate_init=.08)

        paramada = self.doClassifyCrossValidation(X, Y, ada)
        paramknn = self.doClassifyCrossValidation(X, Y, knn)
        paramnb = self.doClassifyCrossValidation(X, Y, nivebase)
        paramdt = self.doClassifyCrossValidation(X, Y, dt)
        paramlr = self.doClassifyCrossValidation(X, Y, lr)
        paramsvm = self.doClassifyCrossValidation(X, Y, svclassifier)
        paramrf = self.doClassifyCrossValidation(X, Y, randomforest)
        parammlp = self.doClassifyCrossValidation(X, Y, mlp)

        result = pd.DataFrame({'Evaluating Function': self.evaluationName,
                               'Ada': paramada.to_numpy()[:, 1],
                               'KNN': paramknn.to_numpy()[:, 1],
                               'NB': paramnb.to_numpy()[:, 1],
                               'DT': paramdt.to_numpy()[:, 1],
                               'LR': paramlr.to_numpy()[:, 1],
                               'SVM': paramsvm.to_numpy()[:, 1],
                               'RF': paramrf.to_numpy()[:, 1],
                               'MLP': parammlp.to_numpy()[:, 1]})
        return result
    def RemoveCorrelated(self, dataframe):
        corr_matrix = dataframe.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = {}
        for i in range(upper.values.shape[0]):
            for j in range(i + 1, upper.values.shape[0]):
                if upper.values[i, j] >= 0.70:
                    to_drop[upper.columns[j]] = 1

        uncorrelated_data = dataframe.drop(to_drop.keys(), axis=1)
        return uncorrelated_data
    def ReadExcelFile(self,filename,sheetname):

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
        return X,y
    def DoClassifyAllFeature(self,path):
        onlyfiles = glob.glob(path)
        cross_totalAUC = pd.DataFrame(
            {'Features': [''], 'Ada': [0], 'Knn': [0], 'NB': [0], 'DT': [0], 'LR': [0], 'SVM': [0], 'RF': [0],
             'MLP': [0]})
        cross_totalACC = pd.DataFrame(
            {'Features': [''], 'Ada': [0], 'Knn': [0], 'NB': [0], 'DT': [0], 'LR': [0], 'SVM': [0], 'RF': [0],
             'MLP': [0]})
        cross_totalRecall = pd.DataFrame(
            {'Features': [''], 'Ada': [0], 'Knn': [0], 'NB': [0], 'DT': [0], 'LR': [0], 'SVM': [0], 'RF': [0],
             'MLP': [0]})
        cross_totalprecision = pd.DataFrame(
            {'Features': [''], 'Ada': [0], 'Knn': [0], 'NB': [0], 'DT': [0], 'LR': [0], 'SVM': [0], 'RF': [0],
             'MLP': [0]})
        cross_totalf1 = pd.DataFrame(
            {'Features': [''], 'Ada': [0], 'Knn': [0], 'NB': [0], 'DT': [0], 'LR': [0], 'SVM': [0], 'RF': [0],
             'MLP': [0]})
        cross_totalmatt = pd.DataFrame(
            {'Features': [''], 'Ada': [0], 'Knn': [0], 'NB': [0], 'DT': [0], 'LR': [0], 'SVM': [0], 'RF': [0],
             'MLP': [0]})


        jack_totalAUC = pd.DataFrame(
            {'Features': [''], 'Ada': [0], 'Knn': [0], 'NB': [0], 'DT': [0], 'LR': [0], 'SVM': [0], 'RF': [0],
             'MLP': [0]})
        jack_totalACC = pd.DataFrame(
            {'Features': [''], 'Ada': [0], 'Knn': [0], 'NB': [0], 'DT': [0], 'LR': [0], 'SVM': [0], 'RF': [0],
             'MLP': [0]})
        jack_totalRecall = pd.DataFrame(
            {'Features': [''], 'Ada': [0], 'Knn': [0], 'NB': [0], 'DT': [0], 'LR': [0], 'SVM': [0], 'RF': [0],
             'MLP': [0]})
        jack_totalprecision = pd.DataFrame(
            {'Features': [''], 'Ada': [0], 'Knn': [0], 'NB': [0], 'DT': [0], 'LR': [0], 'SVM': [0], 'RF': [0],
             'MLP': [0]})
        jack_totalf1 = pd.DataFrame(
            {'Features': [''], 'Ada': [0], 'Knn': [0], 'NB': [0], 'DT': [0], 'LR': [0], 'SVM': [0], 'RF': [0],
             'MLP': [0]})
        jack_totalmatt = pd.DataFrame(
            {'Features': [''], 'Ada': [0], 'Knn': [0], 'NB': [0], 'DT': [0], 'LR': [0], 'SVM': [0], 'RF': [0],
             'MLP': [0]})
        for i, item in enumerate(onlyfiles):
            sheetname = os.path.basename(item).split('.')[0]
            print('processing {}'.format(sheetname))
            X,y = self.ReadExcelFile(item,"Sheet1")
            if (i==0):
                indices = np.arange(X.shape[0])
                X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, indices, test_size=0.2,random_state=2)
            X_train = X[idx_train,:]
            y_train = y[idx_train]
            X_test= X[idx_test,:]
            y_test= y[idx_test]

            result = self.applyAllmodelCrossValidation(X_train,y_train)

            cross_totalACC.loc[i] = [sheetname] + list(result.values[2, 1:])
            cross_totalAUC.loc[i] = [sheetname] + list(result.values[5, 1:])
            cross_totalRecall.loc[i] = [sheetname] + list(result.values[3, 1:])
            cross_totalprecision.loc[i] = [sheetname] + list(result.values[0, 1:])
            cross_totalf1.loc[i] = [sheetname] + list(result.values[1, 1:])
            cross_totalmatt.loc[i] = [sheetname] + list(result.values[4, 1:])

            result = self.applyAllmodelTrainAndTest(X_train, y_train,X_test,y_test)
            jack_totalACC.loc[i] = [sheetname] + list(result.values[2, 1:])
            jack_totalAUC.loc[i] = [sheetname] + list(result.values[5, 1:])
            jack_totalRecall.loc[i] = [sheetname] + list(result.values[3, 1:])
            jack_totalprecision.loc[i] = [sheetname] + list(result.values[0, 1:])
            jack_totalf1.loc[i] = [sheetname] + list(result.values[1, 1:])
            jack_totalmatt.loc[i] = [sheetname] + list(result.values[4, 1:])
            #result.to_csv("result\EEE.{}.csv".format(sheetname))

        cross_totalACC.to_csv("result/Cross.ACC.csv")
        cross_totalAUC.to_csv("result/Cross.AUC.csv")
        cross_totalRecall.to_csv("result/Cross.Recall.csv")
        cross_totalprecision.to_csv("result/Cross.Precision.csv")
        cross_totalf1.to_csv("result/Cross.F1.csv")
        cross_totalmatt.to_csv("result/Cross.Matt.csv")

        jack_totalACC.to_csv("result/Jack.ACC.csv")
        jack_totalAUC.to_csv("result/Jack.AUC.csv")
        jack_totalRecall.to_csv("result/Jack.Recall.csv")
        jack_totalprecision.to_csv("result/Jack.Precision.csv")
        jack_totalf1.to_csv("result/Jack.Jack.F1.csv")
        jack_totalmatt.to_csv("result/Jack.Matt.csv")

feature_xlsx_path = "BenckMarkData/All/xlsx/*.xlsx"

moonlight_object = Moonlihg_Classifier()
moonlight_object.DoClassifyAllFeature(feature_xlsx_path)


