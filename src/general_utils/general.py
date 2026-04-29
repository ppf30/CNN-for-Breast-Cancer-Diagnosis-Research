import os
import pickle
import to
from torch.utils.data import DataLoader

class General():

    def __init__():
        pass

    @staticmethod
    def ensure_dir(path: str)->None:
        """
            Function aimed to check path's 
            existance
            
            Params:
                path(str): The expected path

            Returns:
                None
        """
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def serialize_data(data:object, path:str)->None:
        """ 
            Function aimed to serialize data 

            Params:
                data(object): The data object to store
                path(str): The  path to store data

            Returns:
                None

        """
        # Serialize using pickle
        pickle.dump(obj = data, file = path)


    @staticmethod
    def recover_data(path:str)->DataLoader:
        """ 
            Function aimed to recover serialized
            data

            Params:
                path(str): The path where data is stored

            Returns:
                data(DataLoader): The data object 
        """

        # Get data from path
        try:
            data = pickle.load(file = path)
            return data
        except FileNotFoundError as e:
            print(f'Exception({e})')

        



