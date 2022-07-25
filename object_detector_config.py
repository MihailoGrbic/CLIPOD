import json

class CLIPObjectDetectorConfig:
    def __init__(self, **config_dict):
        # Fill the config with default values
        self.device = 'cpu'
        self.batch_size = 8

        self.init_strat = 'segments'
        self.num_segments = 5
        self.repeat_wo_best = False
        self.init_threshold = 1e-1
        
        self.text_strat = 'one-negative'
        self.negative_text = 'background'
        self.num_other_cats = 10

        self.bbox_strat = 'random'
        self.crop=True
        self.num_tries=100
        
        # Fill with values from 
        self.__dict__.update(config_dict)

    @classmethod
    def from_file(cls, file_path):
        with open(file_path) as f:
            data = f.read()
        config_dict = json.loads(data)
        
        return cls(config_dict)
