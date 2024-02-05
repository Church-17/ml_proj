from scipy.io import arff
from scipy.stats import zscore
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter.ttk import *

def start_analisys2(self):
        self.window2 = tk.Toplevel()
        self.window2.geometry('280x210')
        self.title:str = "INFORMAZIONI SUL DATASET"
        self.window2.title(self.title)

        self.data_frame = Frame(self.window2)
        self.data_frame.columnconfigure((0, 1), weight=1)
        self.data_frame.grid(row=0, column=1, sticky=tk.EW)

        df =self.dataset
        X=df.iloc[:,:-1] 
        y=df.iloc[:,-1].to_numpy()[:,]

        balance_perc = (sum(y != b'1'))/(y.size) * 100
        null_values = X.isnull().sum().sum()
        non_null_values = X.count().sum().sum()
        null_perc = (null_values / ((null_values + non_null_values))) * 100
        num_outliers = len(X[(np.abs(X.select_dtypes(include=[np.number]).apply(zscore)) > 3).any(axis=1)])
        outlier_perc = (num_outliers / len(X)) * 100
        min_std = X.std().min()
        max_std = X.std().max()
        mean_std = X.std().mean()

        Label(self.data_frame, text="Data Analisys:", font=("Helvetica", 12)).grid(row=0, column=0, sticky=tk.W)
        Label(self.data_frame, text=f'Percentuale di bilanciamento: {round(balance_perc,5)}%', font=("Helvetica", 10)).grid(row=1, column=0, sticky=tk.W)
        Label(self.data_frame, text=f'Numero di valori nulli: {round(null_values,5)}', font=("Helvetica", 10)).grid(row=2, column=0, sticky=tk.W)
        Label(self.data_frame, text=f'Numero di valori non nulli: {round(non_null_values,5)}', font=("Helvetica", 10)).grid(row=3, column=0, sticky=tk.W)
        Label(self.data_frame, text=f'Percentuale di valori nulli: {round(null_perc,5)}%', font=("Helvetica", 10)).grid(row=4, column=0, sticky=tk.W)
        Label(self.data_frame, text=f'Numero di outlier: {round(num_outliers,5)}', font=("Helvetica", 10)).grid(row=5, column=0, sticky=tk.W)
        Label(self.data_frame, text=f'Percentuale di outlier: {round(outlier_perc,5)}%', font=("Helvetica", 10)).grid(row=6, column=0, sticky=tk.W)
        Label(self.data_frame, text=f'Deviazione standard minima: {round(min_std,5)}', font=("Helvetica", 10)).grid(row=7, column=0, sticky=tk.W)
        Label(self.data_frame, text=f'Deviazione standard massima: {round(max_std,5)}', font=("Helvetica", 10)).grid(row=8, column=0, sticky=tk.W)
        Label(self.data_frame, text=f'Deviazione standard media: {round(mean_std,5)}', font=("Helvetica", 10)).grid(row=9, column=0, sticky=tk.W)

