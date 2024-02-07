from scipy.stats import zscore
import numpy as np
import tkinter as tk
from tkinter.ttk import *
from dataset import split_attrib_class

attr_list = [f"Attr{i+1}" for i in range(64)]   # Name of attributes from the dataset

def data_analisys(dataset):
    "Shows a tool for dataset analysis"
    
    # Initializing the window
    window2 = tk.Toplevel()
    window2.geometry('500x500')
    window2.resizable(False, False)
    title:str = "Info dataset"
    window2.title(title)
    window2.columnconfigure(0, weight=1)

    data_frame = Frame(window2)
    data_frame.columnconfigure((0, 1), weight=1)
    data_frame.grid(row=0, column=0, pady=(0, 20), sticky=tk.NSEW)

    desctiption_frame = Frame(window2)
    desctiption_frame.columnconfigure(0, weight=1)
    desctiption_frame.grid(row=1, column=0, pady=(0, 5), sticky=tk.NSEW)

    X, y = split_attrib_class(dataset)

    # Computing Dataset Statistics
    balance_perc = (sum(y != b'1'))/(y.size) * 100
    null_values = X.isnull().sum().sum()
    non_null_values = X.count().sum().sum()
    null_perc = (null_values / ((null_values + non_null_values))) * 100
    num_outliers = len(X[(np.abs(X.select_dtypes(include=[np.number]).apply(zscore)) > 3).any(axis=1)])
    outlier_perc = (num_outliers / len(X)) * 100
    min_std = X.std().min()
    max_std = X.std().max()
    mean_std = X.std().mean()

    # Widgets for Dataset Statistics
    Label(data_frame, text="Dataset Statistics:", font=("Helvetica", 12)).grid(row=0, column=0, columnspan=2)
    Label(data_frame, text=f'Balance proportion:', font=("Helvetica", 10)).grid(row=1, column=0, padx=4)
    Label(data_frame, text=f'Null values:', font=("Helvetica", 10)).grid(row=2, column=0, padx=4)
    Label(data_frame, text=f'non-Null values:', font=("Helvetica", 10)).grid(row=3, column=0, padx=4)
    Label(data_frame, text=f'Null values proportion:', font=("Helvetica", 10)).grid(row=4, column=0, padx=4)
    Label(data_frame, text=f'Outliers:', font=("Helvetica", 10)).grid(row=5, column=0, padx=4)
    Label(data_frame, text=f'Outliers proportion:', font=("Helvetica", 10)).grid(row=6, column=0, padx=4)
    Label(data_frame, text=f'Minimum standard deviation:', font=("Helvetica", 10)).grid(row=7, column=0, padx=4)
    Label(data_frame, text=f'Mean standard deviation', font=("Helvetica", 10)).grid(row=8, column=0, padx=4)
    Label(data_frame, text=f'Maximum standard deviation:', font=("Helvetica", 10)).grid(row=9, column=0, padx=4)
    Label(data_frame, text=f'{round(balance_perc,5)}%', font=("Helvetica", 10)).grid(row=1, column=1, padx=4, sticky=tk.W)
    Label(data_frame, text=f'{round(null_values,5)}', font=("Helvetica", 10)).grid(row=2, column=1, padx=4, sticky=tk.W)
    Label(data_frame, text=f'{round(non_null_values,5)}', font=("Helvetica", 10)).grid(row=3, column=1, padx=4, sticky=tk.W)
    Label(data_frame, text=f'{round(null_perc,5)}%', font=("Helvetica", 10)).grid(row=4, column=1, padx=4, sticky=tk.W)
    Label(data_frame, text=f'{round(num_outliers,5)}', font=("Helvetica", 10)).grid(row=5, column=1, padx=4, sticky=tk.W)
    Label(data_frame, text=f'{round(outlier_perc,5)}%', font=("Helvetica", 10)).grid(row=6, column=1, padx=4, sticky=tk.W)
    Label(data_frame, text=f'{round(min_std,5)}', font=("Helvetica", 10)).grid(row=7, column=1, padx=4, sticky=tk.W)
    Label(data_frame, text=f'{round(max_std,5)}', font=("Helvetica", 10)).grid(row=8, column=1, padx=4, sticky=tk.W)
    Label(data_frame, text=f'{round(mean_std,5)}', font=("Helvetica", 10)).grid(row=9, column=1, padx=4, sticky=tk.W)

    # Widget for Feature Analisys
    Label(desctiption_frame, text="Feature Analysis:", font=("Helvetica", 12)).grid(row=0, column=0, pady=4)
    Label(desctiption_frame, text="Select the feature", font=("Helvetica", 10)).grid(row=1, column=0, pady=4)
    attr_combo = Combobox(desctiption_frame, width=10, state='readonly')
    attr_combo['values'] = attr_list
    attr_combo.grid(row=2, column=0)

    # Printing statistics of a feature
    print_stats = lambda event : Label(desctiption_frame, text=dataset[attr_combo.get()].describe()).grid(row=3, column=0, pady=4)

    attr_combo.bind("<<ComboboxSelected>>", print_stats)
