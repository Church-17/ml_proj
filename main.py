import tkinter as tk
from tkinter.ttk import *

class ML_Project_GUI:
    def __init__(self, dataset):
        self.title = "Machine Learning Project"
        self.dataset = dataset
        self.setup_window()
        self.window.mainloop()

    def setup_window(self):
        self.window = tk.Tk()
        self.window.geometry("700x400")
        self.window.title(self.title)
        self.window.focus_force()




if __name__ == "__main__":
    dataset = None # Open csv
    ML_Project_GUI(dataset)