import tkinter as tk
from tkinter.ttk import *
from classification import *
from ROC import draw_roc_curve

def destroy_child(frame:Frame):
    for widget in frame.winfo_children():
        widget.destroy()
Frame.destroy_child = destroy_child
 
sampling_tuple = ('No sampling', 'Random without replacement', 'Random with replacement', 'Fixed stratified', 'Proportional stratified')
balancing_tuple = ('No balancing', 'Random undersampling', 'Probabilistic undersampling', 'Nearest to nearest', 'Nearest to farthest', 'Cluster Centroid', 'Random oversampling', 'Oversampling SMOTE', 'Oversampling ADASYN', 'Hybrid SMOTEENN')
reduction_tuple = ('No dimensionality reduction', 'Principal Components Analysis', 'Sparse Random Projection', 'Gaussian Random Projection', 'Feature Agglomeration')
transformation_tuple = ('No transformation', 'Z-Score standardization', 'Min-Max standardization', 'L1 normalization', 'L2 normalization')

class ML_Project_GUI:
    def __init__(self, dataset):
        self.title:str = "Machine Learning Project"
        self.dataset = dataset
        self.classifier_picked = None
        self.setup_window()

    # Initialize window
    def setup_window(self):
        # Window
        self.window = tk.Tk()
        self.window.title(self.title)
        self.window.focus_force()
        self.window.geometry("500x450")
        self.window.minsize(500, 450)
        self.window.iconbitmap("icon.ico")
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(1, weight=1)

        # Preprocessing frame
        self.preproc_frame = Frame(self.window)
        self.preproc_title = Label(self.preproc_frame, text="Data preprocessing", font=("Helvetica", 12))
        self.sampling_title = Label(self.preproc_frame, text="Sampling")
        self.balancing_title = Label(self.preproc_frame, text="Balancing")
        self.reduction_title = Label(self.preproc_frame, text="Dimensionality reduction")
        self.transformation_title = Label(self.preproc_frame, text="Transformation")
        self.sampling_combo = Combobox(self.preproc_frame, state='readonly')
        self.sampling_combo['values'] = sampling_tuple
        self.sampling_combo.current(0)
        self.balancing_combo = Combobox(self.preproc_frame, state='readonly')
        self.balancing_combo['values'] = balancing_tuple
        self.balancing_combo.current(0)
        self.reduction_combo = Combobox(self.preproc_frame, state='readonly')
        self.reduction_combo['values'] = reduction_tuple
        self.reduction_combo.current(0)
        self.transformation_combo = Combobox(self.preproc_frame, state='readonly')
        self.transformation_combo['values'] = transformation_tuple
        self.transformation_combo.current(0)
        
        self.preproc_frame.grid(row=0, column=0, sticky=tk.EW)
        self.preproc_frame.columnconfigure((0, 1), weight=1)
        self.preproc_title.grid(row=0, column=0, columnspan=2, pady=8)
        self.sampling_title.grid(row=1, column=0, padx=20, pady=(5, 0))
        self.balancing_title.grid(row=1, column=1, padx=20, pady=(5, 0))
        self.reduction_title.grid(row=3, column=0, padx=20, pady=(5, 0))
        self.transformation_title.grid(row=3, column=1, padx=20, pady=(5, 0))
        self.sampling_combo.grid(row=2, column=0, padx=20, pady=2, sticky=tk.EW)
        self.balancing_combo.grid(row=2, column=1, padx=20, pady=2, sticky=tk.EW)
        self.reduction_combo.grid(row=4, column=0, padx=20, pady=2, sticky=tk.EW)
        self.transformation_combo.grid(row=4, column=1, padx=20, pady=2, sticky=tk.EW)
        
        # Classifier frame
        self.classifier_frame = Frame(self.window)
        self.classifier_title = Label(self.classifier_frame, text="Classifier", font=("Helvetica", 12))
        self.classifier_combo = Combobox(self.classifier_frame, width=40, state='readonly')
        self.classifier_combo['values'] = classifier_tuple
        self.classifier_combo.bind("<<ComboboxSelected>>", self.classifier_options)
        self.classifier_option_frame = Frame(self.classifier_frame)
        
        self.classifier_frame.grid(row=1, column=0, sticky=tk.N+tk.EW)
        self.classifier_frame.columnconfigure(0, weight=1)
        self.classifier_title.grid(row=0, column=0, pady=(20, 7))
        self.classifier_combo.grid(row=1, column=0)
        self.classifier_option_frame.grid(row=2, column=0, padx=20, pady=10, sticky=tk.EW)

        # Result frame
        self.result_frame = Frame(self.window)
        self.result_title = Label(self.result_frame, text="Performance", font=("Helvetica", 12))
        self.accuracy_label = Label(self.result_frame, text="Accuracy")
        self.sensibility_label = Label(self.result_frame, text="Sensibility")
        self.specificity_label = Label(self.result_frame, text="Specificity")
        self.accuracy = Label(self.result_frame)
        self.sensibility = Label(self.result_frame)
        self.specificity = Label(self.result_frame)

        self.result_frame.grid(row=2, column=0, sticky=tk.S+tk.EW, pady=5)
        self.result_frame.columnconfigure(tuple(range(3)), weight=1)
        self.result_title.grid(row=0, column=0, columnspan=3, pady=7)
        self.accuracy_label.grid(row=1, column=0)
        self.sensibility_label.grid(row=1, column=1)
        self.specificity_label.grid(row=1, column=2)
        self.accuracy.grid(row=2, column=0)
        self.sensibility.grid(row=2, column=1)
        self.specificity.grid(row=2, column=2)

        # Bottom frame
        self.bottom_frame = Frame(self.window)
        self.progressbar = Progressbar(self.bottom_frame, mode='determinate', length=400)
        self.notify_label = Label(self.bottom_frame, font=("Helvetica", 11))
        self.start_button = Button(self.bottom_frame, text="Start", state=tk.DISABLED, command=self.start_classification)
        self.roc_button = Button(self.bottom_frame, text="ROC curve", state=tk.DISABLED, command=self.roc_curve)
        self.close_button = Button(self.bottom_frame, text="Close", command=self.window.destroy)

        self.bottom_frame.grid(row=3, column=0, padx=10, pady=10, sticky=tk.EW)
        self.bottom_frame.columnconfigure(0, weight=1)
        self.progressbar.grid(row=0, column=0, columnspan=4, padx=5, pady=5)
        self.notify_label.grid(row=1, column=0, padx=5, pady=5)
        self.start_button.grid(row=1, column=1, padx=5, pady=5)
        self.roc_button.grid(row=1, column=2, padx=5, pady=5)
        self.close_button.grid(row=1, column=3, padx=5, pady=5)

    def classifier_options(self, event):
        selected_option = self.classifier_combo.get()
        if self.classifier_picked == selected_option:
            return
        
        self.classifier_picked = selected_option
        self.start_button.config(state=tk.ACTIVE) # Enable training
        self.notify_label.config(text="Classifier chosen", foreground="green")
        self.classifier_option_frame.destroy_child()

        if selected_option == 'Ensamble classifier':
            self.classifier_option_frame.columnconfigure(tuple(range(3)), weight=1)

            # Radiobutton for hard or soft voting
            self.voting_var = tk.IntVar()
            self.voting_frame = Frame(self.classifier_option_frame)
            self.hardvoting_rb = Radiobutton(self.voting_frame, text="Hard voting", variable=self.voting_var, value=0)
            self.softvoting_rb = Radiobutton(self.voting_frame, text="Soft voting", variable=self.voting_var, value=1)
            self.voting_frame.grid(row=0, column=0)
            self.hardvoting_rb.grid(row=0, column=0, sticky=tk.W)
            self.softvoting_rb.grid(row=1, column=0, sticky=tk.W)

            # Radiobutton for weight or no
            self.weight_var = tk.IntVar()
            self.weight_frame = Frame(self.classifier_option_frame)
            self.unweighted_rb = Radiobutton(self.weight_frame, text="Non weighted", variable=self.weight_var, value=0, command=self.weight_options)
            self.weighted_rb = Radiobutton(self.weight_frame, text="Weighted", variable=self.weight_var, value=1, command=self.weight_options)
            self.weight_frame.grid(row=0, column=1)
            self.unweighted_rb.grid(row=0, column=0, sticky=tk.W)
            self.weighted_rb.grid(row=1, column=0, sticky=tk.W)

            # Spinbox for weights
            self.weights_frame = Frame(self.classifier_option_frame)
            self.weights_label = Label(self.weights_frame, text="Weights:")
            self.weight1 = Spinbox(self.weights_frame, from_=0, to=100, width=4, state=tk.DISABLED)
            self.weight2 = Spinbox(self.weights_frame, from_=0, to=100, width=4, state=tk.DISABLED)
            self.weight3 = Spinbox(self.weights_frame, from_=0, to=100, width=4, state=tk.DISABLED)
            self.weight1.set(1)
            self.weight2.set(1)
            self.weight3.set(1)
            self.weights_frame.grid(row=0, column=2)
            self.weights_label.grid(row=0, column=0, columnspan=3)
            self.weight1.grid(row=1, column=0, padx=4)
            self.weight2.grid(row=1, column=1, padx=4)
            self.weight3.grid(row=1, column=2, padx=4)

        elif(selected_option == 'K nearest neighbor'):
            self.classifier_option_frame.columnconfigure(tuple(range(2)), weight=1)
            self.classifier_option_frame.columnconfigure(tuple(range(2, 3)), weight=0)
            
            # Spinbox for K
            self.k_label = Label(self.classifier_option_frame, text="K value:")
            self.k_spinbox = Spinbox(self.classifier_option_frame, from_=1, to=100, width=4)
            self.k_spinbox.set(1)
            self.k_label.grid(row=0, column=0)
            self.k_spinbox.grid(row=1, column=0)

            # Combobox for distance function
            self.distance_label = Label(self.classifier_option_frame, text="Distance:")
            self.distance_combobox = Combobox(self.classifier_option_frame, state='readonly')
            self.distance_combobox['values'] = distance_tuple
            self.distance_combobox.current(0)
            self.distance_label.grid(row=0, column=1)
            self.distance_combobox.grid(row=1, column=1)

        elif(selected_option == 'Decision tree'): 
            self.classifier_option_frame.columnconfigure(tuple(range(2)), weight=1)
            self.classifier_option_frame.columnconfigure(tuple(range(2, 3)), weight=0)

            # Radiobutton for pre or post pruning
            self.prune_var = tk.IntVar()
            self.prune_frame = Frame(self.classifier_option_frame)
            self.pre_pruning_rb = Radiobutton(self.prune_frame, text="Pre-pruning", variable=self.prune_var, value=0)
            self.post_pruning_rb = Radiobutton(self.prune_frame, text="Post-Pruning", variable=self.prune_var, value=1)
            self.prune_frame.grid(row=0, column=0)
            self.pre_pruning_rb.grid(row=0, column=0, sticky=tk.W)
            self.post_pruning_rb.grid(row=1, column=0, sticky=tk.W)

            # Combobox for purity function
            self.purity_frame = Frame(self.classifier_option_frame)
            self.purity_label = Label(self.purity_frame, text="Purity:")
            self.purity_combobox = Combobox(self.purity_frame, state='readonly')
            self.purity_combobox['values'] = purity_tuple
            self.purity_combobox.current(0)
            self.purity_frame.grid(row=0, column=1)
            self.purity_label.grid(row=0, column=0)
            self.purity_combobox.grid(row=1, column=0)

    def weight_options(self): # managing weight selection
        if self.weight_var.get() == 0:
            self.weight1.config(state=tk.DISABLED)
            self.weight2.config(state=tk.DISABLED)
            self.weight3.config(state=tk.DISABLED)
        else:
            self.weight1.config(state=tk.ACTIVE)
            self.weight2.config(state=tk.ACTIVE)
            self.weight3.config(state=tk.ACTIVE)

    def start_classification(self):
        start_classification(self.classifier_picked)

    def roc_curve(self):
        draw_roc_curve(self.y_test, self.y_pred_prob)
