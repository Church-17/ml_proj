import tkinter as tk
from tkinter.ttk import *
from tkinter import IntVar, StringVar


class ML_Project_GUI:
    def __init__(self, dataset):
        self.title = "Machine Learning Project"
        self.dataset = dataset
        self.setup_window()

    # Initialize window
    def setup_window(self):
        # Window
        self.window = tk.Tk()
        self.window.title(self.title)
        self.window.focus_force()
        self.window.geometry("500x500")
        self.window.minsize(400, 400)
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(1, weight=1)

        # Preprocessing frame
        self.preproc_frame = Frame(self.window)
        self.preproc_title = Label(self.preproc_frame, text="Data Preprocessing", font=("Helvetica", 12))
        self.sampling_title = Label(self.preproc_frame, text="Sampling")
        self.balancing_title = Label(self.preproc_frame, text="Balancing")
        self.reduction_title = Label(self.preproc_frame, text="Dimensionality reduction")
        self.transformation_title = Label(self.preproc_frame, text="Transformation")
        self.sampling_combo = Combobox(self.preproc_frame)
        self.balancing_combo = Combobox(self.preproc_frame)
        self.reduction_combo = Combobox(self.preproc_frame)
        self.transformation_combo = Combobox(self.preproc_frame)
        
        self.preproc_frame.grid(row=0, column=0, sticky=tk.EW)
        self.preproc_frame.columnconfigure(0, weight=1)
        self.preproc_frame.columnconfigure(1, weight=1)
        self.preproc_title.grid(row=0, column=0, columnspan=2, pady=8)
        self.sampling_title.grid(row=1, column=0, padx=20, pady=(5,0))
        self.balancing_title.grid(row=1, column=1, padx=20, pady=(5,0))
        self.reduction_title.grid(row=3, column=0, padx=20, pady=(5,0))
        self.transformation_title.grid(row=3, column=1, padx=20, pady=(5,0))
        self.sampling_combo.grid(row=2, column=0, padx=20, pady=2, sticky=tk.EW)
        self.balancing_combo.grid(row=2, column=1, padx=20, pady=2, sticky=tk.EW)
        self.reduction_combo.grid(row=4, column=0, padx=20, pady=2, sticky=tk.EW)
        self.transformation_combo.grid(row=4, column=1, padx=20, pady=2, sticky=tk.EW)
        
        # Classifier frame
        self.classifier_frame = Frame(self.window)
        self.classifier_title = Label(self.classifier_frame, text="Classifier", font=("Helvetica", 12))
        self.classifier_combo = Combobox(self.classifier_frame, width=60) #l'ho aumentato un po, da 40 a 60
        self.classifier_combo['values'] = ('Nessuna Selezione', 'Classificatore Multiplo', 'Alberi Decisionali', 'KNN')
        self.classifier_combo.current(0)
        self.classifier_combo.bind("<<ComboboxSelected>>", self.classifier_selected)
        
        self.classifier_frame.grid(row=1, column=0, columnspan=3, sticky=tk.N+tk.EW)
        self.classifier_frame.columnconfigure(0, weight=1)
        self.classifier_frame.columnconfigure(1, weight=1)
        self.classifier_frame.columnconfigure(2, weight=3)
        self.classifier_title.grid(row=0, column=0, columnspan=3, pady=(20,7))
        self.classifier_combo.grid(row=1, column=0, columnspan=3)

        # Bottom frame
        self.bottom_frame = Frame(self.window)
        self.notify_label = Label(self.bottom_frame, font=("Segoe UI", 10))
        self.start_button = Button(self.bottom_frame, text="Start", state=tk.DISABLED)
        self.roc_button = Button(self.bottom_frame, text="ROC curve", state=tk.DISABLED)
        self.close_button = Button(self.bottom_frame, text="Close", command=self.window.destroy)

        self.bottom_frame.grid(row=4, column=0, padx=10, pady=12, sticky=tk.EW)
        self.bottom_frame.columnconfigure(0, weight=1)
        self.notify_label.grid(row=0, column=0, padx=5, pady=5)
        self.start_button.grid(row=0, column=1, padx=5, pady=5)
        self.roc_button.grid(row=0, column=2, padx=5, pady=5)
        self.close_button.grid(row=0, column=3, padx=5, pady=5)


    def classifier_selected(self, event):
        # Distrugge i widget se esistono
        if hasattr(self, 'classifier_multi'):
            self.classifier_multi.destroy()
        if hasattr(self, 'hardvoting_cb'):
            self.hardvoting_cb.destroy()
        if hasattr(self, 'softvoting_cb'):
            self.softvoting_cb.destroy()
        if hasattr(self, 'weighted_cb'):
            self.weighted_cb.destroy()
        if hasattr(self, 'unweighted_cb'):
            self.unweighted_cb.destroy()
        if hasattr(self, 'weights_label'):
            self.weights_label.destroy()
        if hasattr(self, 'weights_frame'):
            self.weights_frame.destroy()
        if hasattr(self, 'weight1'):
            self.weight1.destroy()
        if hasattr(self, 'weight2'):
            self.weight2.destroy()
        if hasattr(self, 'weight3'):
            self.weight3.destroy()
        if hasattr(self, 'k_label'):
            self.k_label.destroy()
        if hasattr(self, 'k_spinbox'):
            self.k_spinbox.destroy()
        if hasattr(self, 'post_pruning_cb'):
            self.post_pruning_cb.destroy()
        if hasattr(self, 'pre_pruning_cb'):
            self.pre_pruning_cb.destroy()
        selected_option = self.classifier_combo.get()
        if selected_option == 'Classificatore Multiplo':
            self.classifier_multi = Frame(self.window)
            self.classifier_multi.grid(row=3, column=0, pady=10, padx = 10, sticky=tk.EW)
            self.classifier_multi.columnconfigure(0, weight=1)
            self.classifier_multi.columnconfigure(1, weight=1)
            self.classifier_multi.columnconfigure(2, weight=1)

            # Checkbox per "hardvoting" e "softvoting"
            self.vote_var = StringVar(value="none")
            self.hardvoting_rb = Radiobutton(self.classifier_multi, text="hardvoting", variable=self.vote_var, value="hardvoting")
            self.softvoting_rb = Radiobutton(self.classifier_multi, text="softvoting", variable=self.vote_var, value="softvoting")
            self.hardvoting_rb.grid(row=0, column=0, sticky='W')
            self.softvoting_rb.grid(row=1, column=0, sticky='W')

            # Radiobutton per "pesato" e "non pesato"
            self.weight_var = StringVar(value="none")
            self.weighted_rb = Radiobutton(self.classifier_multi, text="pesato", variable=self.weight_var, value="pesato")
            self.unweighted_rb = Radiobutton(self.classifier_multi, text="non pesato", variable=self.weight_var, value="non pesato")
            self.weighted_rb.grid(row=0, column=1, sticky='W')
            self.unweighted_rb.grid(row=1, column=1, sticky='W')


            # Label e spinbox per "pesi"
            self.weights_label = Label(self.classifier_multi, text="pesi:")
            self.weights_label.grid(row=0, column=2)
            self.weights_frame = Frame(self.classifier_multi)
            self.weights_frame.grid(row=1, column=2)

            self.weight1 = Spinbox(self.weights_frame, from_=0, to=100, width=4)
            self.weight1.set(1)
            self.weight2 = Spinbox(self.weights_frame, from_=0, to=100, width=4)
            self.weight2.set(1)
            self.weight3 = Spinbox(self.weights_frame, from_=0, to=100, width=4)
            self.weight3.set(1)

            self.weight1.pack(side="left", padx=5)
            self.weight2.pack(side="left", padx=5)
            self.weight3.pack(side="left", padx=5)

        elif(selected_option == 'KNN'):
            self.classifier_multi = Frame(self.window)
            self.classifier_multi.grid(row=3, column=0, pady=10, padx=10, sticky=tk.EW)
            self.classifier_multi.columnconfigure(0, weight=1)
            self.classifier_multi.columnconfigure(1, weight=1)

            self.k_label = Label(self.classifier_multi, text="Valore di K:")
            self.k_label.grid(row=0, column=0, sticky='W')
            self.k_spinbox = Spinbox(self.classifier_multi, from_=1, to=100, width=4)
            self.k_spinbox.set(1)
            self.k_spinbox.grid(row=0, column=1, sticky='W') 

        elif(selected_option == 'Alberi Decisionali'): 
            self.classifier_multi = Frame(self.window)
            self.classifier_multi.grid(row=3, column=0, pady=10, padx=10, sticky=tk.EW)
            self.classifier_multi.columnconfigure(0, weight=1)

            # Variabile per i Radiobutton
            self.prune_var = StringVar(value="none")

            # Radiobutton per "Post-Pruning" e "Pre-pruning"
            self.post_pruning_rb = Radiobutton(self.classifier_multi, text="Post-Pruning", variable=self.prune_var, value="post")
            self.pre_pruning_rb = Radiobutton(self.classifier_multi, text="Pre-pruning", variable=self.prune_var, value="pre")
            self.post_pruning_rb.grid(row=0, column=0, sticky='W')
            self.pre_pruning_rb.grid(row=1, column=0, sticky='W')
        else:
            pass



def main():
    dataset = None # Open csv
    GUI = ML_Project_GUI(dataset)
    GUI.window.mainloop()


if __name__ == "__main__":
    main()