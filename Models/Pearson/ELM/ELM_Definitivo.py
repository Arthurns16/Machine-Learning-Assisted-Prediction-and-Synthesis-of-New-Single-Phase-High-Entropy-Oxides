from sklearn.datasets import load_iris, load_digits, load_diabetes, make_regression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import elm
import pandas as pd
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold,ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold,ShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score 
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
import random
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import random
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np
import pickle




def dados_treino_e_validacao() :
    dataset = pd.read_excel('Input_Total_filtrado.xlsx')
    dataset = dataset.loc[:,dataset.apply(pd.Series.nunique) != 1]

    # Divide o conjunto de dados em X e y
    X = dataset.iloc[:, 2:206].values 
    encoder = LabelEncoder()
    encoder.fit(dataset.iloc[:,211].values)
    y =  encoder.fit_transform(dataset.iloc[:,211].values)
    scaler = StandardScaler()
    scaler.fit(X)
    X= scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    with open('X_train.pickle', 'wb') as f:
        pickle.dump(X_train, f)

    with open('X_test.pickle', 'wb') as f:
        pickle.dump(X_test, f)

    with open('y_train.pickle', 'wb') as f:
        pickle.dump(y_train, f)

    with open('y_test.pickle', 'wb') as f:
        pickle.dump(y_test, f)

    with open('X.pickle', 'wb') as f:
        pickle.dump(X, f)

    with open('y.pickle', 'wb') as f:
        pickle.dump(y, f)

    with open('scaler.pickle', 'wb') as f:
        pickle.dump(scaler, f)

    with open('encoder.pickle', 'wb') as f:
        pickle.dump(encoder, f)



#dados_treino_e_validacao()

def reduzir_dimensao (reduzendo, todas_as_features, colunas_para_manter) :

  arr = reduzendo

  # obter todos os rótulos das colunas do dataframe
  rotulos_das_colunas = todas_as_features.columns.tolist()

  # criar uma lista com as posições das colunas a serem deletadas
  colunas_para_deletar = [i for i in range(len(rotulos_das_colunas)) if rotulos_das_colunas[i] not in colunas_para_manter]

  # deletar as colunas do array numpy
  arr = np.delete(arr, colunas_para_deletar, axis=1)

  return arr

def hiperparametros_treinoEML_metricas_cv() :


    # Carregar as variáveis
    with open('X_train.pickle', 'rb') as f:
        X_train = pickle.load(f)

    with open('X_test.pickle', 'rb') as f:
        X_test = pickle.load(f)

    with open('y_train.pickle', 'rb') as f:
        y_train = pickle.load(f)

    with open('y_test.pickle', 'rb') as f:
        y_test = pickle.load(f)

    with open('X.pickle', 'rb') as f:
        X = pickle.load(f)

    with open('y.pickle', 'rb') as f:
        y = pickle.load(f)

    with open('scaler.pickle', 'rb') as f:
        scaler = pickle.load(f)

    with open('encoder.pickle', 'rb') as f:
        encoder = pickle.load(f)
    
    ############################

    dataset = pd.read_excel('Input_Total.xlsx')
    dataset = dataset.loc[:,dataset.apply(pd.Series.nunique) != 1]
    dataset = dataset.filter(regex=r'^(?!.*Keq)')
    dataset = dataset.drop('Classe', axis=1)
    dataset = dataset.drop('Composto', axis=1)
    dataset = dataset.drop('Átomos', axis=1)
    
    dataframe_pandas_features= dataset
    
    
    colunas_para_manter = ['atomic_ea_minimo', 'atomic_ea_maximo', 'atomic_ea_soma',
       'atomic_en_allen _soma', 'atomic_en_allen _desvio',
       'atomic_en_allredroch_minimo', 'atomic_hatm_minimo',
       'atomic_spacegroupnum_maximo', 'atomic_spacegroupnum_desvio',
        'van_der_waals_rad_minimo']
    
    X_train = reduzir_dimensao(X_train,dataframe_pandas_features,colunas_para_manter).copy()
    X_test = reduzir_dimensao(X_test,dataframe_pandas_features,colunas_para_manter).copy()
    
    
    ##################
        
    param_grid = {
        'hidden_units': [3,6,9,12,15,18,20,22,25,32,36,40,50,75,100,150,200,400],
        'activation_function' : ['sigmoid', 'relu', 'sin', 'tanh','leaky_relu'],
        'C' : [0,1,2,3,4,5,6,7,8,9,10,13,17,25],
        'random_type': ['normal','uniform'],
        'treinador' : ['no_re', 'solution1' , 'solution2']
    }

    melhor_r2_score = -99999999.9
    melhor_numero = None
    melhor_funcao = None
    melhor_c = None
    melhor_tipo = None
    melhor_treinador = None 
    model = None

    # Create a 2-fold cross validation
    kf = KFold(n_splits=2)

    for numero in param_grid['hidden_units']:
        for funcao in param_grid['activation_function']:
            for cl in param_grid['C']:
                for tipo in param_grid['random_type']:
                    for treinador in param_grid['treinador'] :
                    
                    
                        
                        if 1<2:


                            r2_scores = []
                            for train_index, test_index in kf.split(X_train):
                                try:
                                    x_train_fold, x_val_fold = X_train[train_index], X_train[test_index]
                                    y_train_fold, y_val_fold = y_train[train_index], y_train[test_index]
                                   
                                
                                      
                                    model = elm.elm(hidden_units=numero, 
                                                activation_function=funcao, random_type=tipo, 
                                                C=cl, elm_type='clf',x=x_train_fold, y=y_train_fold,one_hot=True)
                                                
                                    model.fit(treinador)
                                    # predict on test data
                                    prediction = model.predict(x_val_fold)
                                    # calculate r2 score
                                    r2 = r2_score(y_val_fold, prediction)
                                    r2_scores.append(r2)
                                    print(r2)

                                    # calculate average r2 score
                                    avg_r2 = np.mean(r2_scores)

                                    if melhor_r2_score < avg_r2 and avg_r2<1 and avg_r2>0 :
                                        melhor_r2_score = avg_r2
                                        melhor_numero = numero
                                        melhor_funcao = funcao
                                        melhor_c = cl
                                        melhor_tipo = tipo
                                        melhor_treinador = treinador
                                except:
                                    pass
    
    
    with open("hiperparametrosEML_CV.txt", "w") as arquivo:
        arquivo.write('hidden_units = '+ str(melhor_numero) + ',' + 'activation_function = '+ '\''+str(melhor_funcao)+ '\''
       + ',' + 'C='+str(melhor_c)+ ','+ 'random_type=' +'\''+str(melhor_tipo)+ '\'' 
       +',' + 'x=x_train, y=y_train,elm_type=\'clf\''+
       ' treinador= '+'\''+str(melhor_treinador)+'\'' + ' melhor_r2_score = '+ str(melhor_r2_score))

#hiperparametros_treinoEML_metricas_cv()

def gerar_modelo ():

    # Carregar as variáveis
    with open('X_train.pickle', 'rb') as f:
        X_train = pickle.load(f)

    with open('X_test.pickle', 'rb') as f:
        X_test = pickle.load(f)

    with open('y_train.pickle', 'rb') as f:
        y_train = pickle.load(f)

    with open('y_test.pickle', 'rb') as f:
        y_test = pickle.load(f)

    with open('X.pickle', 'rb') as f:
        X = pickle.load(f)

    with open('y.pickle', 'rb') as f:
        y = pickle.load(f)

    with open('scaler.pickle', 'rb') as f:
        scaler = pickle.load(f)

    with open('encoder.pickle', 'rb') as f:
        encoder = pickle.load(f)
        


    ############################

    dataset = pd.read_excel('Input_Total.xlsx')
    dataset = dataset.loc[:,dataset.apply(pd.Series.nunique) != 1]
    dataset = dataset.filter(regex=r'^(?!.*Keq)')
    dataset = dataset.drop('Classe', axis=1)
    dataset = dataset.drop('Composto', axis=1)
    dataset = dataset.drop('Átomos', axis=1)
    
    dataframe_pandas_features= dataset
    
    
    colunas_para_manter = ['atomic_ea_minimo', 'atomic_ea_maximo', 'atomic_ea_soma',
       'atomic_en_allen _soma', 'atomic_en_allen _desvio',
       'atomic_en_allredroch_minimo', 'atomic_hatm_minimo',
       'atomic_spacegroupnum_maximo', 'atomic_spacegroupnum_desvio',
        'van_der_waals_rad_minimo']
    
    X_train = reduzir_dimensao(X_train,dataframe_pandas_features,colunas_para_manter).copy()
    X_test = reduzir_dimensao(X_test,dataframe_pandas_features,colunas_para_manter).copy()
    
    
    ##################
    
        
    print(len(X_train[0]))
    
    model = elm.elm(hidden_units=25, activation_function='tanh',
    random_type='normal', x=X_train, y=y_train, C=3, elm_type='clf',one_hot=True) 
    beta, train_accuracy, running_time = model.fit('solution2')


    # test - Observe que estamos calculando a acurácia com y_test ainda codificado
    prediction = model.predict(X_test)
    prediction_decoded = encoder.inverse_transform(prediction)
    print("classifier test prediction:", prediction_decoded)
    print('classifier test accuracy:', model.score(X_test, y_test))
    
    with open('modelEML.pickle', 'wb') as f:
        pickle.dump(model, f)

#gerar_modelo()


# Intervalo de confiança da acurácia 

def new_boot(X_test, y_test):
  novo_X = X_test.copy()

  novo_Y = y_test.copy()

  c= 0
  while c < len(X_test):
    rand = random.randint(0,len(X_test)-1)
    novo_X[c] = X_test[rand].copy()
    novo_Y[c] = y_test[rand]

    c= c+1


  return novo_X , novo_Y

# Pega a acurácia de um boot
# Ajuste em pegar a mérica sem ser pelo report
def pegar_acuracia_do_relatorio(novo_X , novo_Y):

  return accuracy_score(novo_Y, model.predict(novo_X), sample_weight=None)






if 1<2:
    # Carregar X_train
    with open('X_train.pickle', 'rb') as f:
        X_train = pickle.load(f)

    # Carregar X_test
    with open('X_test.pickle', 'rb') as f:
        X_test = pickle.load(f)

    # Carregar y_train
    with open('y_train.pickle', 'rb') as f:
        y_train = pickle.load(f)

    # Carregar y_test
    with open('y_test.pickle', 'rb') as f:
        y_test = pickle.load(f)

    # Carregar model
    with open('modelELM.pickle', 'rb') as f:
        model = pickle.load(f)

    # Carregar scaler
    with open('scalerELM.pickle', 'rb') as f:
        scaler = pickle.load(f)

    # Carregar encoder
    with open('encoderELM.pickle', 'rb') as f:
        encoder = pickle.load(f)

    dataset = pd.read_excel('Input_Total.xlsx')
    dataset = dataset.loc[:,dataset.apply(pd.Series.nunique) != 1]
    dataset = dataset.filter(regex=r'^(?!.*Keq)')
    dataset = dataset.drop('Classe', axis=1)
    dataset = dataset.drop('Composto', axis=1)
    dataset = dataset.drop('Átomos', axis=1)
    
    dataframe_pandas_features= dataset
    
    
    colunas_para_manter = ['atomic_ea_minimo', 'atomic_ea_maximo', 'atomic_ea_soma',
       'atomic_en_allen _soma', 'atomic_en_allen _desvio',
       'atomic_en_allredroch_minimo', 'atomic_hatm_minimo',
       'atomic_spacegroupnum_maximo', 'atomic_spacegroupnum_desvio',
        'van_der_waals_rad_minimo']
    
    X_train = reduzir_dimensao(X_train,dataframe_pandas_features,colunas_para_manter).copy()
    X_test = reduzir_dimensao(X_test,dataframe_pandas_features,colunas_para_manter).copy()
    
    
    print(model.score(X_test, y_test))
    print(accuracy_score(y_test, model.predict(X_test), sample_weight=None))
    
    # calcula a distribuição dos boots

    numero_boots = 40001
    lista_boots = []
    contador = 0

    while contador < numero_boots:
      x, y = new_boot(X_test, y_test)
      lista_boots.append(pegar_acuracia_do_relatorio(x, y))
      contador =contador +1

    plt.hist(lista_boots)
    plt.xlabel('Accuracy')
    plt.ylabel('Number of Bootstrap Samples')
    plt.savefig('intervalo_confianca_ELM.png')
    
    with open("lista_boots_ELM.pkl", "wb") as file:
        pickle.dump(lista_boots, file)

    #converte a lista em float explicitamente para a função percentile ser aplicada
    array = list()
    for elemento in lista_boots:
      array.append(float(elemento))

    # calcula os limites da integral da gaussiana que correspondem a área desejada

    alpha = 0.95
    p = ((1.0-alpha)/2.0) * 100
    lower =  np.percentile(array, p)
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper =  np.percentile(array, p)

    print("Intervalo de confiança : ["+str(lower)+","+str(upper)+"]")
    print("Acurácia 'real' do modelo performada no teste : "+ str(accuracy_score(y_test, model.predict(X_test), sample_weight=None)))



    # Supondo que 'model' é o seu modelo treinado e 'X_test', 'y_test' são seus dados de teste
    # Primeiro, você precisa fazer previsões usando o seu conjunto de teste
    y_pred = model.predict(X_test)

    # Gere a matriz de confusão
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Obtenha os nomes das classes
    class_names = ['Fluorita','Mixed' ,'Monoclinica' ,'Perovskita',
     'Pirocloro' ,'RockSalt' ,'Spinel']

    # Para melhor visualização, você pode usar o Seaborn para plotar a matriz de confusão
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()
    

    print(encoder.inverse_transform(y_pred)) 
    print(encoder.inverse_transform(y_test)) 
    print(y_test) 
 







    # Supondo que y_test já esteja disponível e não seja binário
    # Se y_test já for binário (em formato one-hot), você pode pular esta etapa
    # Binarizar os rótulos em uma configuração um contra todos
    classes = ['Fluorita', 'Mixed', 'Monoclinica', 'Perovskita', 'PerovskitaOrto', 'Pirocloro', 'RockSalt', 'Spinel']

    classes_ing = [
    "Fluorite",
    "Mixed",
    "Monoclinic",
    "Perovskite",
    "Orthorhombic Perovskite",
    "Pyrochlore",
    "RockSalt",
    "Spinel"
]  

    y_test_binarized = label_binarize(encoder.inverse_transform(y_test), classes=classes)
    n_classes = y_test_binarized.shape[1]

    # Prever classes
    y_pred = model.predict(X_test)

    # Binarizar as previsões
    y_pred_binarized = label_binarize(encoder.inverse_transform(y_pred), classes=classes)

    # Computar ROC curve e ROC area para cada classe
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_binarized[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Associar os nomes das classes binarizadas
    class_labels = classes

    # Garantir que class_labels corresponde às classes
    assert len(class_labels) == n_classes, "O número de class_labels deve corresponder ao número de classes."

    # Plot da curva ROC para cada classe
    colors = cycle(['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black', 'pink'])
    linestyles = cycle(['-', '--', '-.', ':'])
    plt.figure(figsize=(10, 8))
    for i, (color, linestyle) in zip(range(n_classes), zip(colors, linestyles)):
        plt.plot(fpr[i], tpr[i], color=color, linestyle=linestyle, lw=2, alpha=0.7,
                 label='{0} (AUC = {1:0.2f})'.format(classes_ing[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC one-vs-all')
    plt.legend(loc="lower right")
    plt.show()

    plt.savefig("roc_curves.png")

    with open('fpr_tpr_rocauc_ELM.pkl', 'wb') as file:
        pickle.dump({'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}, file)

