from sklearn.preprocessing import StandardScaler
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
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, \
    GradientBoostingClassifier, VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import random
import glob

class Moonlihg_Classifier():
    def __init__(self):
        self.evaluationName = ['Precision', 'F1Score', 'Accuracy', 'Recall', 'Matt', 'Auc']
        self.runtimes = 35
        self.nfold = 10

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


        return evlP
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

        result = np.zeros((8,6))

        result[0,:] = paramada
        result[1,:] = paramknn
        result[2,:] = paramnb
        result[3,:] = paramdt
        result[4,:] = paramlr
        result[5,:] = paramsvm
        result[6,:] = paramrf
        result[7,:] = parammlp
        return result

    def doClassifyCrossValidation(self, X, y, classifier,kf):
        Data = X
        evlP = np.zeros((self.nfold,6))
        k = 0

        for train_index, test_index in kf.split(Data,y) :
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
        average = evlP.mean(axis=0)
        return average

    def applyAllmodelCrossValidation(self, X, Y,kf):

        ada = AdaBoostClassifier(n_estimators=200, base_estimator=None, learning_rate=0.1, random_state=1)
        knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
        nivebase = GaussianNB()
        dt = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
        lr = LogisticRegression(random_state=0)
        svclassifier = SVC(kernel='rbf', degree=3, C=2)
        randomforest = RandomForestClassifier(n_estimators=200)  # Train the model on training data
        mlp = MLPClassifier(hidden_layer_sizes=(20, 10, 5), max_iter=1000, alpha=1e-4, solver='sgd', verbose=0,tol=1e-4,random_state=1,learning_rate_init=.08)

        paramada = self.doClassifyCrossValidation(X, Y, ada,kf)
        paramknn = self.doClassifyCrossValidation(X, Y, knn,kf)
        paramnb = self.doClassifyCrossValidation(X, Y, nivebase,kf)
        paramdt = self.doClassifyCrossValidation(X, Y, dt,kf)
        paramlr = self.doClassifyCrossValidation(X, Y, lr,kf)
        paramsvm = self.doClassifyCrossValidation(X, Y, svclassifier,kf)
        paramrf = self.doClassifyCrossValidation(X, Y, randomforest,kf)
        parammlp = self.doClassifyCrossValidation(X, Y, mlp,kf)

        result = np.zeros((8, 6))

        result[0,:] = paramada
        result[1,:] = paramknn
        result[2,:] = paramnb
        result[3,:] = paramdt
        result[4,:] = paramlr
        result[5,:] = paramsvm
        result[6,:] = paramrf
        result[7,:] = parammlp


        return result


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

    def ApplyAllData(self,path):
        onlyfiles = glob.glob(path)
        file_count = len(onlyfiles)
        cross_totalAUC = np.zeros((self.runtimes,file_count,8))
        cross_totalACC = np.zeros((self.runtimes,file_count,8))
        cross_totalRecall = np.zeros((self.runtimes,file_count,8))
        cross_totalprecision = np.zeros((self.runtimes,file_count,8))
        cross_totalf1 = np.zeros((self.runtimes,file_count,8))
        cross_totalmatt = np.zeros((self.runtimes,file_count,8))


        jack_totalAUC = np.zeros((self.runtimes,file_count,8))
        jack_totalACC = np.zeros((self.runtimes,file_count,8))
        jack_totalRecall =np.zeros((self.runtimes,file_count,8))
        jack_totalprecision = np.zeros((self.runtimes,file_count,8))
        jack_totalf1 = np.zeros((self.runtimes,file_count,8))
        jack_totalmatt = np.zeros((self.runtimes,file_count,8))

        feature_name = []

        method_name = ['AdaBoost','Knn','Nb','Dt','Lr','SVM','Rf','MLP']

        #pd.DataFrame({'Features': [''], 'Ada': [0], 'Knn': [0], 'NB': [0], 'DT': [0], 'LR': [0], 'SVM': [0], 'RF': [0],
        # 'MLP': [0]})
        for runs in range(self.runtimes):
            print (runs)
            for i, item in enumerate(onlyfiles):
                sheetname = os.path.basename(item).split('.')[0]
                if (runs==0):
                    feature_name.append(sheetname)
                print('processing {}'.format(sheetname))
                X,y = self.ReadExcelFile(item,"Sheet1")
                if (i==0):
                    indices = np.arange(X.shape[0])
                    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, indices, test_size=0.2,random_state=random.randint(1,1000))
                    kf = KFold(n_splits=self.nfold, shuffle=True, random_state=random.randint(1, 1000))

                X_train = X[idx_train,:]
                y_train = y[idx_train]
                X_test= X[idx_test,:]
                y_test= y[idx_test]

                #self.evaluationName = ['Precision', 'F1Score', 'Accuracy', 'Recall', 'Matt', 'Auc']
                result = self.applyAllmodelCrossValidation(X_train,y_train,kf)
                cross_totalprecision[runs,i,:] =result[:,0]
                cross_totalf1[runs,i,:] =result[:,1]
                cross_totalACC[runs,i,:] =result[:,2]
                cross_totalRecall[runs,i,:] =result[:,3]
                cross_totalmatt[runs,i,:] =result[:,4]
                cross_totalAUC[runs,i,:] =result[:,5]



                result = self.applyAllmodelTrainAndTest(X_train, y_train,X_test,y_test)

                jack_totalprecision[runs,i,:]= result[:,0]
                jack_totalf1[runs,i,:]= result[:,1]
                jack_totalACC[runs,i,:]= result[:,2]
                jack_totalRecall[runs,i,:]= result[:,3]
                jack_totalmatt[runs,i,:]= result[:,4]
                jack_totalAUC[runs,i,:]= result[:,5]


        cross_totalprecision_mean, cross_totalprecision_var = self.Mean_Variance(cross_totalprecision)
        cross_totalACC_mean, cross_totalACC_var = self.Mean_Variance(cross_totalACC)
        cross_totalAUC_mean, cross_totalAUC_var = self.Mean_Variance(cross_totalAUC)
        cross_totalRecall_mean, cross_totalRecall_var = self.Mean_Variance(cross_totalRecall)
        cross_totalf1_mean, cross_totalf1_var = self.Mean_Variance(cross_totalf1)
        cross_totalmatt_mean, cross_totalmatt_var = self.Mean_Variance(cross_totalmatt)
#Save mean result
        df1 = pd.DataFrame(cross_totalprecision_mean,index=feature_name,columns=method_name)
        df2 = pd.DataFrame(cross_totalACC_mean,index=feature_name,columns=method_name)
        df3 = pd.DataFrame(cross_totalAUC_mean,index=feature_name,columns=method_name)
        df4 = pd.DataFrame(cross_totalRecall_mean,index=feature_name,columns=method_name)
        df5 = pd.DataFrame(cross_totalf1_mean,index=feature_name,columns=method_name)
        df6 = pd.DataFrame(cross_totalmatt_mean,index=feature_name,columns=method_name)


        with pd.ExcelWriter('Result/Cross_Mean.xlsx') as writer:
            df1.to_excel(writer, sheet_name='Precision')
            df2.to_excel(writer, sheet_name='Accuracy')
            df3.to_excel(writer, sheet_name='Auc')
            df4.to_excel(writer, sheet_name='Recall')
            df5.to_excel(writer, sheet_name='F1')
            df6.to_excel(writer, sheet_name='Matt')

        df1 = pd.DataFrame(cross_totalprecision_var,index=feature_name,columns=method_name)
        df2 = pd.DataFrame(cross_totalACC_var,index=feature_name,columns=method_name)
        df3 = pd.DataFrame(cross_totalAUC_var,index=feature_name,columns=method_name)
        df4 = pd.DataFrame(cross_totalRecall_var,index=feature_name,columns=method_name)
        df5 = pd.DataFrame(cross_totalf1_var,index=feature_name,columns=method_name)
        df6 = pd.DataFrame(cross_totalmatt_var,index=feature_name,columns=method_name)

        with pd.ExcelWriter('Result/Cross_Variance.xlsx') as writer:
            df1.to_excel(writer, sheet_name='Precision')
            df2.to_excel(writer, sheet_name='Accuracy')
            df3.to_excel(writer, sheet_name='Auc')
            df4.to_excel(writer, sheet_name='Recall')
            df5.to_excel(writer, sheet_name='F1')
            df6.to_excel(writer, sheet_name='Matt')

        jack_totalprecision_mean, jack_totalprecision_var = self.Mean_Variance(jack_totalprecision)
        jack_totalACC_mean, jack_totalACC_var = self.Mean_Variance(jack_totalACC)
        jack_totalAUC_mean, jack_totalAUC_var = self.Mean_Variance(jack_totalAUC)
        jack_totalRecall_mean, jack_totalRecall_var = self.Mean_Variance(jack_totalRecall)
        jack_totalf1_mean, jack_totalf1_var = self.Mean_Variance(jack_totalf1)
        jack_totalmatt_mean, jack_totalmatt_var = self.Mean_Variance(jack_totalmatt)

        df1 = pd.DataFrame(jack_totalprecision_mean,index=feature_name,columns=method_name)
        df2 = pd.DataFrame(jack_totalACC_mean,index=feature_name,columns=method_name)
        df3 = pd.DataFrame(jack_totalAUC_mean,index=feature_name,columns=method_name)
        df4 = pd.DataFrame(jack_totalRecall_mean,index=feature_name,columns=method_name)
        df5 = pd.DataFrame(jack_totalf1_mean,index=feature_name,columns=method_name)
        df6 = pd.DataFrame(jack_totalmatt_mean,index=feature_name,columns=method_name)

        with pd.ExcelWriter('Result/Jack_Mean.xlsx') as writer:
            df1.to_excel(writer, sheet_name='Precision')
            df2.to_excel(writer, sheet_name='Accuracy')
            df3.to_excel(writer, sheet_name='Auc')
            df4.to_excel(writer, sheet_name='Recall')
            df5.to_excel(writer, sheet_name='F1')
            df6.to_excel(writer, sheet_name='Matt')

        df1 = pd.DataFrame(jack_totalprecision_var,index=feature_name,columns=method_name)
        df2 = pd.DataFrame(jack_totalACC_var,index=feature_name,columns=method_name)
        df3 = pd.DataFrame(jack_totalAUC_var,index=feature_name,columns=method_name)
        df4 = pd.DataFrame(jack_totalRecall_var,index=feature_name,columns=method_name)
        df5 = pd.DataFrame(jack_totalf1_var,index=feature_name,columns=method_name)
        df6 = pd.DataFrame(jack_totalmatt_var,index=feature_name,columns=method_name)

        with pd.ExcelWriter('Result/Jack_Variance.xlsx') as writer:
            df1.to_excel(writer, sheet_name='Precision')
            df2.to_excel(writer, sheet_name='Accuracy')
            df3.to_excel(writer, sheet_name='Auc')
            df4.to_excel(writer, sheet_name='Recall')
            df5.to_excel(writer, sheet_name='F1')
            df6.to_excel(writer, sheet_name='Matt')


    def Mean_Variance(self,arr):
        mean_ada = np.mean(arr, axis=0)
        var_ada = np.var(arr, axis=0)
        return mean_ada, var_ada


path = "Data/All/xlsx/*.xlsx"
moonlight_object = Moonlihg_Classifier()
moonlight_object.ApplyAllData(path)