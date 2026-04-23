import os

class General():

    def __init__():
        pass

    @staticmethod
    def ensure_model_dir(path: str)->None:
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