import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.utils import resample


class Custom_Ensemble:

    #inizializzazione del classificatore
    def __init__(self):
        pass

    #imposta i parametri scelti in base al tipo di approccio che si vuole utilizzare
    def set_params (self, weights, voting, algorithm): 

        self.algorithm = algorithm

        #in caso si voglia usare una random forest
        if self.algorithm == 'forest':
            self.classificatore_1 = DecisionTreeClassifier(max_depth=None, criterion='entropy', min_samples_leaf=3, min_samples_split=2)
            self.classificatore_2 = DecisionTreeClassifier(max_depth=None, criterion='entropy', min_samples_leaf=3, min_samples_split=2)
            self.classificatore_3 = DecisionTreeClassifier(max_depth=None, criterion='entropy', min_samples_leaf=3, min_samples_split=2)

        #in caso di bagging o voosting o majority voting standard
        else:
            self.classificatore_1 = DecisionTreeClassifier(max_depth=None, criterion='entropy', min_samples_leaf=3, min_samples_split=2)
            self.classificatore_2 = GaussianNB()
            self.classificatore_3 = SVC(probability=True, kernel='rbf', C=1.5, gamma=0.5)  

        #imposta i parametri del classificatore in base ai parametri scelti dall'utente nella GUI
        self.voting = voting #tipolofia di voting (hard o soft)
        self.w = np.array(weights).astype(int) #array contenente i pesi dei classificatori
        self.labels = [0, 1] #label delle classi
        self.wsum =  self.w.sum() #somma dei pesi
        

     
    #addestramento del classificatore
    def fit(self, x, y):

        #nel caso si stai utilizzando il majority voting standard
        if self.algorithm == 'standard':
            self.classificatore_1.fit(x, y) #si addestrano in modo classico i classificatori
            self.classificatore_2.fit(x, y)
            self.classificatore_3.fit(x, y)
            
        #nel caso si stai utilizzando il bagging
        elif self.algorithm == 'bagging':
            x1 = [] # dichiaro le liste con i campioni di boostrap da usare per il bagging
            y1 = [] # e le loro relative etichette

            for _ in range(3): 
                xi, yi = resample(x, y) #creo il campione di bootstrap
                x1.append(xi) #lo appendo alle liste di record
                y1.append(yi)

            self.classificatore_1.fit(x1[0], y1[0]) # addestro i 3 classificatori con i 3 campioni di bootstrap distinti
            self.classificatore_2.fit(x1[1], y1[1])
            self.classificatore_3.fit(x1[2], y1[2])



        #nel caso si stai utilizzando il boosting
        elif self.algorithm == 'boosting':

            sample_weights = np.full(x.shape[0], (1. / x.shape[0])) #inizializzo i pesi ad un valore uniforme

            for clf in [self.classificatore_1, self.classificatore_2, self.classificatore_3]: #scorro i classificatori

                clf.fit(x, y, sample_weight=sample_weights) #lo addestro con i record che possiedono il loro specifico peso

                #uso il classificatore per fare delle predizioni e in base alle sue performance aumento o diminuisco il suo peso nelle previsioni e il peso dei record classificati in modo errato
                y_pred = clf.predict(x) 
                incorrect = (y_pred != y)
                error = np.mean(np.average(incorrect, weights=sample_weights, axis=0))
                clf_weight = np.log((1. - error) / error) + np.log(2.)
                sample_weights *= np.exp(clf_weight * incorrect * ((sample_weights > 0) | (clf_weight > 0)))
                sample_weights /= np.sum(sample_weights)
                

        #nel caso si stai utilizzando una random forest
        if self.algorithm == 'forest':
            x1 = [] #dichiaro le liste dei sottoinsiemi di record con le rispettive label
            y1 = []
            self.feature_sets = [] #dichiaro le liste dei sottoinsiemi di attributi
            for _ in range(3): #per ogni classificatore 

                #calcolo il sottoinsieme di record
                xi, yi = resample(x, y) 
                x1.append(xi)
                y1.append(yi)

                #calcolo il sottoinsieme di feature
                n_features = x.shape[1] 
                subset_size = int(0.85 * n_features) 
                feature_indices = np.random.choice(n_features, size=subset_size, replace=False)
                self.feature_sets.append(feature_indices)

            #addestro gli alberi sui sottoinsiemi dei feature e record
            self.classificatore_1.fit(x1[0][:, feature_indices], y1[0])
            self.classificatore_2.fit(x1[1][:, feature_indices], y1[1])
            self.classificatore_3.fit(x1[2][:, feature_indices], y1[2])



    def predict(self, test_x):

        proba = []  #inizializzo la matrice delle probabilita

        if self.algorithm == 'forest':  #se si sta utilizzando una random forest 
            
            #faccio la previsione considerando solo il sottoinsieme di attirbuti con cui è stato addestrato l'albero
            proba.append(self.classificatore_1.predict_proba(test_x[:, self.feature_sets[0]]))
            proba.append(self.classificatore_2.predict_proba(test_x[:, self.feature_sets[1]]))
            proba.append(self.classificatore_3.predict_proba(test_x[:, self.feature_sets[2]]))
        
        else: #altrimenti calcolo normalmente le probabilita
            proba.append(self.classificatore_1.predict_proba(test_x))
            proba.append(self.classificatore_2.predict_proba(test_x))
            proba.append(self.classificatore_3.predict_proba(test_x))

        pred_y = np.zeros([len(test_x),1]) #inizializzo la lista delle classi predette

        voting = np.zeros([len(test_x),2]) #inizializzo la matrice con i voti dei classificatori 

        for i in range(0, len(test_x)): #per tutti i record del test set
            for j in range(0,3):  #per tutti i classificatori

                if self.voting == 'hard':  #se la tipologia di voto è l'hardvoting

                    #agigungo 1 a voting solo nella colonna corrispondente alla classe con la probabilita predetta piu alta eventualmente moltiplicata per il peso della classe
                    if proba[j][i][1] > proba[j][i][0]:
                        voting[i][1] += (1 * self.w[j] / self.wsum)
                    else:
                        voting[i][0] += (1 * self.w[j]/ self.wsum)
                
                else: #altrimenti si sta usando un soft voting
                    
                    #aggiungo alla colonna di voting corrispondente ad una certa classe la probabilita predetta dai singoli classificatori, eventualmente moltiplicando il voto per il suo peso
                    voting[i][0] += (float(proba[j][i][0]) * float(self.w[j]/ self.wsum))
                    voting[i][1] += (float(proba[j][i][1]) * float(self.w[j]/ self.wsum))

                    
            pred_y[i] = (self.labels[np.argmax(voting[i][:])]) #la classe predetta sara quella con la probabilita piu alta

        return pred_y 
    


    #predice la probabilita di un record di appartenenre ad una certa classe
    def predict_proba(self, test_x):

        proba = [] #inizializzo la matrice delle probabilita

        if self.algorithm == 'forest': #se si sta utilizzando una random forest 
            proba.append(self.classificatore_1.predict_proba(test_x[:, self.feature_sets[0]])) #faccio la previsione considerando solo il sottoinsieme di attirbuti con cui è stato addestrato l'albero
            proba.append(self.classificatore_2.predict_proba(test_x[:, self.feature_sets[1]]))
            proba.append(self.classificatore_3.predict_proba(test_x[:, self.feature_sets[2]]))
        
        else:  #altrimenti calcolo normalmente le probabilita
            proba.append(self.classificatore_1.predict_proba(test_x))
            proba.append(self.classificatore_2.predict_proba(test_x))
            proba.append(self.classificatore_3.predict_proba(test_x))


        pred_y = np.zeros([len(test_x),2]) #inizializzo la lista delle classi predette

        voting = np.zeros([len(test_x), 2]) #inizializzo la matrice con i voti dei classificatori 

        for i in range(0, len(test_x)): #per tutti i record del test set
            for j in range(0,3): #per tutti i classificatori

                #aggiungo alla colonna di voting corrispondente ad una certa classe la probabilita predetta dai singoli classificatori, eventualmente moltiplicando il voto per il suo peso
                voting[i][0] += (float(proba[j][i][0]) * float(self.w[j]/ self.wsum)) 
                voting[i][1] += (float(proba[j][i][1]) * float(self.w[j]/ self.wsum))  

            pred_y[i][0] = voting[i][0]
            pred_y[i][1] = voting[i][1]
            
        return pred_y #restituisco le predizioni


