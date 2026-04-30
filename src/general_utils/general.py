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
                bool: True if exists and None if it is created

        """

        if not os.path.exists(path):
            os.makedirs(path)
        else:
            print(f"The dir ({path}) exists")
            return True
        
    
    @staticmethod
    def path_builder(base:str, name:str)->str:
        """     
            Function aimed to build a full path

            Params:
                base(str): The base path
                name(str): The filename

            Returns:
                final_str(str): Composed address
        """

        return os.path.join(base, name)


    @staticmethod
    def serialize_data(data:object, path:str, name:str)->None:
        """ 
            Function aimed to serialize data 

            Params:
                data(object): The data object to store
                path(str): The base path to store data
                name(str): The specific filename
            Returns:
                None

        """
        # Check extension
        if len(name.split('.'))==1:
            name+='.pkl'
        
        # Create full path
        full_path = General.path_builder(path, name )

        # Ensure dir existance
        General.ensure_dir(path)

        #Make sure the file is not already there
        if not os.path.exists(full_path):
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
                name(str): The name of the specific file where data is

            Returns:
                data(DataLoader): The data object 
        """
        # Check extension
        if len(name.split('.'))==1:
            name+='.pkl'

        # Create the full path
        full_path = General.path_builder(path, name)

        # Get data from path
        try:
            with open(full_path, mode = "rb") as file:
                data = pickle.load(file)
            return data
        
        except FileNotFoundError as e:
            print(f'Exception({e})')

        



