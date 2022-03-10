import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from sklearn.impute import KNNImputer
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

def manejo_missing_values(train_set, dataset_a_rellenar):
    
    #Comenzamos rellenando estas tres features que presentan un alto % de missings y están relacionadas.
    columnas_a_rellenar = ['horas_de_sol','nubosidad_tarde', 'nubosidad_temprano']
    
    #Añadimos columna para identificar si eran missings originalmente o no
    for c in columnas_a_rellenar:
        dataset_a_rellenar[c+'_era_missing'] = 0
        dataset_a_rellenar.loc[dataset_a_rellenar[c].isnull(),c+'_era_missing'] = 1
    
    #Rellenamos hs de sol de acuerdo al valor promedio que toma entre todas las instancias que comparten misma nubosidad
    for i in np.sort(dataset_a_rellenar.nubosidad_tarde.unique())[1:]:
        promedio_hs_de_sol = train_set[ (train_set.horas_de_sol != np.nan) & (train_set.nubosidad_tarde == i)].horas_de_sol.mean()

        dataset_a_rellenar.loc[(dataset_a_rellenar.horas_de_sol == np.nan) & (dataset_a_rellenar.nubosidad_tarde == i), 
        'horas_de_sol'] = promedio_hs_de_sol

    for i in np.sort(dataset_a_rellenar.nubosidad_temprano.unique())[1:]:
        promedio_hs_de_sol = train_set[ (train_set.horas_de_sol != np.nan) & (train_set.nubosidad_tarde == i)].horas_de_sol.mean()
        
        dataset_a_rellenar.loc[(dataset_a_rellenar.horas_de_sol == np.nan) & (dataset_a_rellenar.nubosidad_tarde == i),
        'horas_de_sol'] = promedio_hs_de_sol
    
    
    #Para valores extremos de nubosidad rellenamos usando el análisis realizado en la primera parte del TP
    for i in [0,1,7,8]:
        promedio_nubosidad_tarde = round(train_set[(train_set.nubosidad_temprano == i) & 
                                        (train_set.nubosidad_tarde != np.nan)].nubosidad_tarde.mean())
        
        promedio_nubosidad_temprano = round(train_set[(train_set.nubosidad_tarde == i) & 
                                        (train_set.nubosidad_temprano != np.nan)].nubosidad_temprano.mean())
        
        dataset_a_rellenar.loc[(dataset_a_rellenar.nubosidad_temprano == i) & 
                               (dataset_a_rellenar.nubosidad_tarde == np.nan),'nubosidad_tarde'] = promedio_nubosidad_tarde
        dataset_a_rellenar.loc[(dataset_a_rellenar.nubosidad_tarde == i) & 
                               (dataset_a_rellenar.nubosidad_temprano == np.nan),'nubosidad_temprano'] = promedio_nubosidad_temprano
    
    #Para el resto, usamos el KNN Imputer    
    imputer = KNNImputer(n_neighbors=4)
    imputer.fit(train_set[columnas_a_rellenar])
    imputer_values = imputer.transform(dataset_a_rellenar[columnas_a_rellenar]) 
    
    dataset_a_rellenar['horas_de_sol'] = imputer_values[:,0].round(1)
    dataset_a_rellenar[['nubosidad_tarde', 'nubosidad_temprano']] = imputer_values[:,1:].round(0)
    
    #Para mm_lluvia_dia rellenamos con la moda que es 0 (valor fuertemente predominante)
    dataset_a_rellenar['mm_lluvia_dia'].fillna(value=0, inplace=True)
    
    #Para el resto, rellenamos con la media
    features_numericos = dataset_a_rellenar.select_dtypes(include=['float64']).columns
    for i in features_numericos:
        if (i not in ['horas_de_sol','nubosidad_temprano','nubosidad_tarde','mm_lluvia_dia']):
            dataset_a_rellenar[i].fillna(value=train_set[i].mean(), inplace=True)

    return dataset_a_rellenar


def normalizar_entre_0_y_1(train_set, set_a_normalizar):
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(train_set)
    x_normalizado = min_max_scaler.transform(set_a_normalizar)
    
    return x_normalizado


def normalizar_segun_maximo_valor_absoluto(train_set, set_a_normalizar):
    abs_scaler = preprocessing.MaxAbsScaler()
    abs_scaler.fit(train_set)
    x_normalizado = abs_scaler.transform(set_a_normalizar)
    
    return x_normalizado


def aplicarPCA(train_set, set_a_reducir, cant_componentes=None):
    pca = PCA(cant_componentes)
    pca.fit(train_set)
    componentes_principales = pca.transform(set_a_reducir)
    
    return pca, componentes_principales
    
    
def estandarizar(train_set, set_a_estandarizar):
    for column in set_a_estandarizar:
        if abs(train_set[column].std()) > 0:
            set_a_estandarizar[column] -= train_set[column].mean()
            set_a_estandarizar[column] /= train_set[column].std()
    
    return set_a_estandarizar



def aplicarOneHot(dataset):
    features_categoricos = obtener_features_categoricos(dataset)
    with_dummies = pd.get_dummies(data=dataset, columns=features_categoricos, dummy_na=True, drop_first=True)
    
    if ('dia' in dataset.columns.tolist()):
        with_dummies.drop(columns=['dia'], axis=1, inplace=True)
    
    return with_dummies



def es_estacion(fechas, mes_inicio, mes_fin):
    if(mes_fin == 0):
        mes_fin = 12
        
    return (((mes_inicio%12 +1) <= fechas.month < mes_fin) or 
            (fechas.month == mes_fin and fechas.day < 21)   or 
            (fechas.month == mes_inicio and fechas.day >= 21))

def agregar_feature_estacion(dataset):
    dataset['dia'] = pd.to_datetime(dataset['dia'], errors='coerce')
    estaciones = []
    for i in range(3,13,3):
        estaciones.append(dataset.dia.apply(es_estacion, args=(i, (i+3)%12)))

    serie_estaciones = pd.Series(data= estaciones, index=['Otonio','Invierno','Primavera','Verano'])
    for j in range(0,4):
        dataset.loc[serie_estaciones[j],'estacion'] = serie_estaciones.index[j]
        
    return dataset


def obtener_features_continuos(dataset):
    features_continuos = dataset.select_dtypes(include=['float64']).columns.tolist()
    features_continuos.remove('nubosidad_temprano')
    features_continuos.remove('nubosidad_tarde')
    
    return features_continuos


def obtener_features_discretos(dataset):
    return ['nubosidad_temprano','nubosidad_tarde','horas_de_sol_era_missing', 
            'nubosidad_tarde_era_missing', 'nubosidad_temprano_era_missing']
    

def obtener_features_categoricos(dataset):
    features_categoricos = dataset.select_dtypes(include=['object']).columns.tolist()
    if('dia' in features_categoricos):
        features_categoricos.remove('dia')
    
    return features_categoricos


def filtrar_features_por_varianza(train_set, set_a_reducir, threshold):
    cols_con_varianza = train_set.var().index.values
    df = train_set[cols_con_varianza].copy()

    selector = VarianceThreshold(threshold=threshold)
    vt = selector.fit(df)
    set_reducido = set_a_reducir.loc[:, cols_con_varianza[vt.get_support()]]
    
    return set_reducido