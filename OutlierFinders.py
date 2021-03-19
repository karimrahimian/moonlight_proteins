from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
import numpy as np
import pandas as pd
import os
from sklearn.metrics import precision_score,recall_score,f1_score,matthews_corrcoef,accuracy_score,roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import random
import glob
import seaborn as sns

class OutlierFinder():
    def __init__(self):
        self.evaluationName = ['Precision', 'F1Score', 'Accuracy', 'Recall', 'Matt', 'Auc']

    def multiclass_roc_auc_score(self,y_test, y_pred, average="macro"):
        lb = LabelBinarizer()
        lb.fit(y_test)
        y_test = lb.transform(y_test)
        y_pred = lb.transform(y_pred)
        return roc_auc_score(y_test, y_pred, average=average)
    def MisClassifyFrequency(self,ytest,ypred,idx):
        mis=[]
        for i in range(len(ytest)):
            if (ytest[i]!=ypred[i]):
                mis.append(idx[i])
        return mis
    def doClassify(self,X,y,classifier,nfold=10):
        Data = X
        misclassify = np.zeros(len(y))
        evlP = np.zeros((nfold,6))
        k=0
        kf = KFold(n_splits=nfold,  shuffle=True,random_state=random.randint(1,100))
        for train_index, test_index in kf.split(Data):
            classifier.fit(Data[train_index],y[train_index])
            y_pred = classifier.predict(Data[test_index])
            y_test = y[test_index]
    
            evlP[k][0]=(precision_score(y_test, y_pred, average='micro'))
            evlP[k][1]=(f1_score(y_test, y_pred, average='macro'))
            evlP[k][2]=(accuracy_score(y_test,y_pred))
            evlP[k][3]=(recall_score(y_test,y_pred, average = "weighted"))
            evlP[k][4]=(matthews_corrcoef(y_test,y_pred))
            evlP[k][5]=self.multiclass_roc_auc_score(y_test,y_pred)
            k+=1
            mis = self.MisClassifyFrequency(y_test,y_pred,test_index)
            for item in mis:
                misclassify[item]+=1
        average =  np.matrix(evlP)
        average = average.mean(axis = 0 )
        average =  np.squeeze(np.asarray(average))
        modelparams = pd.DataFrame({'Evaluating Function':self.evaluationName,'Values':average})
        return modelparams,misclassify

    def applyModel(self,X,y,model):
        misclassify = []
        for i in range(len(y)):
            misclassify.append(0)
            
        for i in range(100):
            params,mis= self.doClassify(X,y,model)
            for i,item in enumerate(mis):
                if (item>=1):
                    misclassify[i]+=1
        return np.array(misclassify)
    def RemoveCorrelated(self,dataframe):
        corr_matrix = dataframe.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = {}
        for i in range(upper.values.shape[0]):
            for j in range(i + 1, upper.values.shape[0]):
                if upper.values[i, j] >= 0.70:
                    to_drop[upper.columns[j]] = 1

        uncorrelated_data = dataframe.drop(to_drop.keys(), axis=1)
        return uncorrelated_data
    def FindCandidateOutlierProteins(self,missing,thresuld):
        indexes = []
        for i,item in enumerate(missing):
            if (item>thresuld):
                indexes.append(i)
        return indexes
    def ReadExcelFile(self, filename, sheetname):

        dataframe = pd.read_excel(io=filename, sheet_name=sheetname)
        uncorrelated = self.RemoveCorrelated(dataframe)
        #       nrow = uncorrelated.values.shape[0]
        #       colheader = list(uncorrelated.columns.values)
        rawdata = dataframe.to_numpy()
        PID = rawdata[:, 0]

        ncol = uncorrelated.values.shape[1]
        rawdata = np.array(uncorrelated.to_numpy())

        X = rawdata[:, 1:ncol - 1]
        X = X.astype(np.float)
        y = rawdata[:, ncol - 1]

        y = y.astype(np.float)
        X = RobustScaler().fit_transform(X)
        return X, y,PID
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
    def FindCOP(self,path,thresould=90):
        onlyfiles = glob.glob(path)
        TotalMissing = set()

        for i,item in enumerate(onlyfiles):
            sheetname = os.path.basename(item).split('.')[0]
            print ('processing {}'.format(sheetname))
            X,y , PID =self.ReadExcelFile(item,"Sheet1")
            ada, knn, nivebase, dt, lg, svclassifier, randomforest, mlp = self.MakeModel()

            misknn = obj.applyModel(X,y, knn)
            knnindex = self.FindCandidateOutlierProteins(misknn,thresould)

            misnb = obj.applyModel(X,y, nivebase)
            nbindex = self.FindCandidateOutlierProteins(misnb,thresould)

            missvm = obj.applyModel(X, y, svclassifier)
            svmindex = self.FindCandidateOutlierProteins(missvm,thresould)

            result = pd.DataFrame({
                                   'KNN': pd.Series(PID[knnindex]),
                                   'LabelKNN': pd.Series(y[knnindex]),
                                   'Miss.KNN': pd.Series(misknn[knnindex]),

                                   'NB':  pd.Series(PID[nbindex]),
                                   'LabelNB': pd.Series(y[nbindex]),
                                   'Miss.NB': pd.Series(misnb[nbindex]),

                                   'SVM': pd.Series(PID[svmindex]),
                                   'LabelSVM': pd.Series(y[svmindex]),
                                   'Miss.SVM': pd.Series(missvm[svmindex]),

            })

            result.to_csv("Result\ProteinIndex.{}.csv".format(sheetname))
            if (sheetname=="SAAC"):
                MissingList = MissingList.intersection(set(PID[knnindex]).intersection(set(PID[svmindex])))
            else:
                MissingList=set(PID[nbindex])
        return MissingList
#        TotalMissing = pd.DataFrame({'Missing': list(MissingList)})
#        TotalMissing.to_csv("Result\TotalMissing.csv")
    def FindAllCOP(self,path,thresould=80):
        onlyfiles = glob.glob(path)
        TotalMissing = set()

        for i,item in enumerate(onlyfiles):
            sheetname = os.path.basename(item).split('.')[0]
            print ('processing {}'.format(sheetname))
            X,y , PID =self.ReadExcelFile(item,"Sheet1")
            ada, knn, nivebase, dt, lg, svclassifier, randomforest, mlp = self.MakeModel()
            misada = obj.applyModel(X,y, ada)
            adaindex = self.FindCandidateOutlierProteins(misada,thresould)
            misknn = obj.applyModel(X,y, knn)
            knnindex = self.FindCandidateOutlierProteins(misknn,thresould)
            misnb = obj.applyModel(X,y, nivebase)
            nbindex = self.FindCandidateOutlierProteins(misnb,thresould)
            misdt = obj.applyModel(X,y, dt)
            dtindex  = self.FindCandidateOutlierProteins(misdt,thresould)
            mislg = obj.applyModel(X, y, lg)
            lgindex = self.FindCandidateOutlierProteins(mislg,thresould)
            missvm = obj.applyModel(X, y, svclassifier)
            svmindex = self.FindCandidateOutlierProteins(missvm,thresould)
            misrf = obj.applyModel(X, y, randomforest)
            rfindex = self.FindCandidateOutlierProteins(misrf,thresould)
            mismlp = obj.applyModel(X, y, mlp)
            mlpindex = self.FindCandidateOutlierProteins(mismlp,thresould)

            result = pd.DataFrame({'Ada': pd.Series(PID[adaindex]),
                                   'KNN': pd.Series(PID[knnindex]),
                                   'NB':  pd.Series(PID[nbindex]),
                                   'DT':  pd.Series(PID[dtindex]),
                                   'LR':  pd.Series(PID[lgindex]),
                                   'SVM': pd.Series(PID[svmindex]),
                                   'RF':  pd.Series(PID[rfindex]),
                                   'MLP': pd.Series(PID[mlpindex])})
            result.to_csv("Result\ProteinIndex.{}.csv".format(sheetname))
            MissingList = set(PID[adaindex]).intersection(
                          set(PID[knnindex])).intersection(
                          set(PID[nbindex])).intersection(
                          set(PID[dtindex])).intersection(
                          set(PID[lgindex])).intersection(
                          set(PID[svmindex])).intersection(
                          set(PID[rfindex])).intersection(
                          set(PID[mlpindex]))
            print (MissingList)
            MissingList1 = pd.DataFrame({'Missing': list(MissingList)})
            MissingList1.to_csv("Result\Missing.{}.csv".format(sheetname))
            print("Error")
            if (i==0):
                TotalMissing = MissingList
                MissingList = pd.DataFrame({'Missing': list(MissingList)})
                MissingList.to_csv("Result\Missing.{}.csv".format(sheetname))
            else:
                TotalMissing = TotalMissing.intersection(MissingList)

            #result.to_csv("Result\AfterRemoveMiss.{}.csv".format(sheetname))
        #except:
            import sys
            print(sys.exc_info()[0])
            pass
        TotalMissing = pd.DataFrame({'Missing': TotalMissing})
        TotalMissing.to_csv("Result\TotalMissing.csv")

TotalMissing  = set()
for i in range(1):
    obj = OutlierFinder()
    if (i==0):
        TotalMissing = obj.FindCOP("CopData/*.xlsx")
    else:
        T1 = obj.FindCOP("CopData/*.xlsx")
        TotalMissing = TotalMissing.intersection(T1)
TotalMissing = pd.DataFrame({'Missing': list(TotalMissing)})
TotalMissing.to_csv("Result\TotalMissing1.csv")