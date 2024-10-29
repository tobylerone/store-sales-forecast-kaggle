import pickle

def load_pickle(filename: str) -> any:

    with open(filename, 'rb') as file:
        # Load the object from the file
        obj = pickle.load(file)

    return obj

def save_pickle(obj: any, filename: str) -> None:

    with open(filename, 'rb') as file:
        # Load the object from the file
        pickle.dump(obj, file)
