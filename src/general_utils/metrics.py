import torch

class Metrics():

    def __init__(self, x:torch.Tensor, y:torch.Tensor)->None:
        self.x = x
        self.y = y
        self._metrics:dict[str, float] = {}
        self.total = self.x.shape[0]


    @property
    def metrics(self):
        return self._metrics.copy()

    def __getitem__(self, key:int)->float:
        try:
            metric = self._metrics[key]
            return metric
        except KeyError as e:
            print(f'No matching found with the key ({key})')
            return False

    

    def __str__(self)->str:

        # Main title
        title = 'METRICS SUMMARY'
        final_chain = '*'*len(title) + '\n' + title + '\n' + '*'*len(title) + '\n'

        # Generate the final chain 
        for key, value in self._metrics.items():
            final_chain+=f'·{key.capitalize()}: {round(value.item(), 4)}\n'
        
        return final_chain


    def feed_metrics(self)->None:
        # Compute dice 
        self.compute_dice()
        



    def compute_dice(self)->None:
        """ 
            Function aimed to compute dice 
            function

        """
        # Calculate each part
        intersection = torch.sum((self.x*self.y), dim = (2,3))
    
        denominator = torch.sum(self.x, dim = (2,3)) + torch.sum(self.y, dim = (2,3))
        # Compute dice
        dice = 2*intersection/(denominator + 1e-08)

        # Update metrics dictionary -> sample's mean
        self._metrics['dice'] = dice.mean()
        

