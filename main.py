from gui import ML_Project_GUI

def main():
    dataset = None # Open csv
    GUI = ML_Project_GUI(dataset)
    GUI.window.mainloop()

if __name__ == "__main__":
    main()