from gui import ML_Project_GUI
from dataset import load_dataset

def main():
    dataset = load_dataset("5year.arff") # Open dataset
    GUI = ML_Project_GUI(dataset)
    GUI.window.mainloop()

if __name__ == "__main__":
    main()