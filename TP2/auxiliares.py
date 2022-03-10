import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.impute import KNNImputer

sns.set()

def obtener_dataset_final():
    x = pd.read_csv('Datasets/nuevo.csv')
    return x

def obtener_datasets():
    x = pd.read_csv('Datasets/features.csv')
    y = pd.read_csv('Datasets/target.csv')
    return preparar_datasets(x, y)

def obtener_training_set():
    x = pd.read_csv('Datasets/Training/features.csv')
    y = pd.read_csv('Datasets/Training/target.csv')
    return x,y

def obtener_valdev_set():
    x = pd.read_csv('Datasets/Valdev/features.csv')
    y = pd.read_csv('Datasets/Valdev/target.csv')
    return x,y

def obtener_holdout_set():
    x = pd.read_csv('Datasets/Holdout/features.csv')
    y = pd.read_csv('Datasets/Holdout/target.csv')
    return x,y

def obtener_prediccion_set():
    x = pd.read_csv('Datasets/Prediccion/features.csv')
    return x

def preparar_datasets(x, y):
    dataset = pd.merge(x, y, how='left', left_on='id', right_on='id')
    dataset.drop('id', inplace=True, axis=1)
    dataset = dataset[(dataset.llovieron_hamburguesas_al_dia_siguiente == 'si') | (dataset.llovieron_hamburguesas_al_dia_siguiente == 'no')]
    dataset = dataset[dataset.presion_atmosferica_tarde != '10.167.769.999.999.900']
    dataset = dataset[dataset.presion_atmosferica_tarde != '1.009.555']
    dataset.presion_atmosferica_tarde = dataset.presion_atmosferica_tarde.astype(np.float64)
    dataset['dia'] = pd.to_datetime(dataset['dia'], errors='coerce')
    dataset.reset_index(inplace=True)
    dataset.drop(axis=1, columns='index', inplace=True)
    
    target = dataset['llovieron_hamburguesas_al_dia_siguiente']
    dataset.drop(columns='llovieron_hamburguesas_al_dia_siguiente',axis=1, inplace=True)
    
    return dataset, target


def plot_confusion_matrix(y_true, y_pred):
    fig, ax = plt.subplots(dpi=100)
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, vmin=0, yticklabels=["No lloverán", "Lloverán"], xticklabels=["No lloverán", "lloverán"], ax=ax, fmt='g')
    ax.set_title("Matriz de confusion")
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    
def plot_roc(_fpr, _tpr, x):

    roc_auc = auc(_fpr, _tpr)

    plt.figure(figsize=(15, 10))
    plt.plot(
        _fpr, _tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})'
    )
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
def crear_archivo_predicciones(predicciones : np.array, modelo, id_registro):
    archivo = open("predicciones/"+modelo+".csv", "w")
    archivo.write("id,llovieron_hamburguesas_al_dia_siguiente\n")
    i = 0
    for prediccion in predicciones:
        archivo.write(str(id_registro[i])+ "," + str(prediccion) + "\n")
        i = i + 1
    archivo.close()
    
def graficarHistoriaDeLaRed(historia_red):
    plt.figure(dpi=125, figsize=(7, 2))
    plt.plot(historia_red.history['loss'], label="Training loss")
    plt.plot(historia_red.history['val_loss'], label="Validation loss")
    plt.title('Loss del modelo')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()