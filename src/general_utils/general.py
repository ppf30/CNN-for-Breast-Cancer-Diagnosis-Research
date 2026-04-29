import os
import pickle
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
            print('hola')
            os.makedirs(path)
        else:
            print(f"The dir ({path}) exists")

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

        # Create full path
        full_path = path + 'train_data'

        # Serialize using pickle
        with open(full_path, mode = 'wb') as file:
            pickle.dump(data, file)
        


    @staticmethod
    def recover_data(path:str, name:str)->DataLoader:
        """ 
            Function aimed to recover serialized
            data

            Params:
                path(str): The path where data is stored
                name(str): The name of the specific file where it is

            Returns:
                data(DataLoader): The data object 
        """

        # Create the full path
        full_path = path + name

        # Get data from path
        try:
            with open(full_path, mode = "rb") as file:
                data = pickle.load(file)
            return data
        
        except FileNotFoundError as e:
            print(f'Exception({e})')

        



