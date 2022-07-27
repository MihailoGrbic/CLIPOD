import json

from util import dotdict

class CLIPObjectDetectorConfig:
    def __init__(self, config_dict):
        # Fill the config with default values
        self.batch_size = 8

        self.init_strat = 'segments'                # single or segments
        self.init_settings = dotdict({
            'num_segments' : 4,
            'repeat_wo_best' : False,
            'max_repeat_steps' : 1,
            'threshold' : 1e-1,
        })
        
        self.text_strat = 'one-negative'
        self.negative_text = 'background'
        self.num_other_cats = 10

        self.bbox_strat = 'random'
        self.crop=True
        self.num_tries=100

        self.text_query_prepend = "A photo of a "
        
        # Fill with values from config_dict
        if 'init_settings' in config_dict:
            self.init_settings.update(config_dict['init_settings'])
            del config_dict['init_settings']
        self.__dict__.update(config_dict)

    @classmethod
    def from_file(cls, file_path):
        with open(file_path) as f:
            data = f.read()
        config_dict = json.loads(data)
        
        return cls(config_dict)
