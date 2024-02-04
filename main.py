import tkinter as tk
from tkinter.ttk import *

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
        self.preproc_title.grid(row=0, column=0, columnspan=3, pady=8)
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
        self.classifier_combo = Combobox(self.classifier_frame, width=40)
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

        self.bottom_frame.grid(row=2, column=0, padx=10, pady=12, sticky=tk.EW)
        self.bottom_frame.columnconfigure(0, weight=1)
        self.notify_label.grid(row=0, column=0, padx=5, pady=5)
        self.start_button.grid(row=0, column=1, padx=5, pady=5)
        self.roc_button.grid(row=0, column=2, padx=5, pady=5)
        self.close_button.grid(row=0, column=3, padx=5, pady=5)

    def classifier_selected(self, event):
        pass



def main():
    dataset = None # Open csv
    GUI = ML_Project_GUI(dataset)
    GUI.window.mainloop()


if __name__ == "__main__":
    main()