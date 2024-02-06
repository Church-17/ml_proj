from scipy.stats import zscore
import numpy as np
import tkinter as tk
from tkinter.ttk import *

def data_analisys(self):
        self.window2 = tk.Toplevel()
        self.window2.geometry('300x230')
        self.window2.minsize(300, 230)
        self.window2.maxsize(300, 230)
        self.title:str = "Info dataset"
        self.window2.title(self.title)
        self.window2.columnconfigure(0, weight=1)
        self.window2.rowconfigure(0, weight=1)

        self.data_frame = Frame(self.window2)
        self.data_frame.columnconfigure((0, 1), weight=1)
        # self.data_frame.rowconfigure(tuple(range(10)), weight=1)
        self.data_frame.grid(row=0, column=0, pady=(0, 5))

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

        Label(self.data_frame, text="Data Analisys:", font=("Helvetica", 12)).grid(row=0, column=0, columnspan=2)
        Label(self.data_frame, text=f'Percentuale di bilanciamento:', font=("Helvetica", 10)).grid(row=1, column=0, padx=4)
        Label(self.data_frame, text=f'Numero di valori nulli:', font=("Helvetica", 10)).grid(row=2, column=0, padx=4)
        Label(self.data_frame, text=f'Numero di valori non nulli:', font=("Helvetica", 10)).grid(row=3, column=0, padx=4)
        Label(self.data_frame, text=f'Percentuale di valori nulli:', font=("Helvetica", 10)).grid(row=4, column=0, padx=4)
        Label(self.data_frame, text=f'Numero di outlier:', font=("Helvetica", 10)).grid(row=5, column=0, padx=4)
        Label(self.data_frame, text=f'Percentuale di outlier:', font=("Helvetica", 10)).grid(row=6, column=0, padx=4)
        Label(self.data_frame, text=f'Deviazione standard minima:', font=("Helvetica", 10)).grid(row=7, column=0, padx=4)
        Label(self.data_frame, text=f'Deviazione standard massima:', font=("Helvetica", 10)).grid(row=8, column=0, padx=4)
        Label(self.data_frame, text=f'Deviazione standard media:', font=("Helvetica", 10)).grid(row=9, column=0, padx=4)
        Label(self.data_frame, text=f'{round(balance_perc,5)}%', font=("Helvetica", 10)).grid(row=1, column=1, padx=4, sticky=tk.W)
        Label(self.data_frame, text=f'{round(null_values,5)}', font=("Helvetica", 10)).grid(row=2, column=1, padx=4, sticky=tk.W)
        Label(self.data_frame, text=f'{round(non_null_values,5)}', font=("Helvetica", 10)).grid(row=3, column=1, padx=4, sticky=tk.W)
        Label(self.data_frame, text=f'{round(null_perc,5)}%', font=("Helvetica", 10)).grid(row=4, column=1, padx=4, sticky=tk.W)
        Label(self.data_frame, text=f'{round(num_outliers,5)}', font=("Helvetica", 10)).grid(row=5, column=1, padx=4, sticky=tk.W)
        Label(self.data_frame, text=f'{round(outlier_perc,5)}%', font=("Helvetica", 10)).grid(row=6, column=1, padx=4, sticky=tk.W)
        Label(self.data_frame, text=f'{round(min_std,5)}', font=("Helvetica", 10)).grid(row=7, column=1, padx=4, sticky=tk.W)
        Label(self.data_frame, text=f'{round(max_std,5)}', font=("Helvetica", 10)).grid(row=8, column=1, padx=4, sticky=tk.W)
        Label(self.data_frame, text=f'{round(mean_std,5)}', font=("Helvetica", 10)).grid(row=9, column=1, padx=4, sticky=tk.W)
