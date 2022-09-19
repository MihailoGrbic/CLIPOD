import json

from util import dotdict

class CLIPObjectDetectorConfig:
    def __init__(self, config_dict):
        
        # Fill the config with default values
        self.batch_size = 64
        self.text_prompt_prepend = "A photo of a "
        self.crop = True
        
        self.init_strat = 'segments'                # single or segments
        self.init_settings = dotdict({
            'num_segments' : 4,
            'repeat_wo_best' : False,
            'max_repeat_steps' : 1,
            'threshold' : 1.805e-1,
        })
        
        self.detection_text_strat = 'promising-cats'
        self.detection_text_settings = dotdict({
            'num_other_cats' : 100,
        })

        self.detection_strat = 'random'
        self.detection_settings = dotdict({
            'num_tries' : 100,
            'nms_thresh' : 0.3,
        })


        # Fill with values from config_dict
        config_dict = config_dict.copy()
        if 'init_settings' in config_dict:
            self.init_settings.update(config_dict['init_settings'])
            del config_dict['init_settings']
        if 'detection_text_settings' in config_dict:
            self.detection_text_settings.update(config_dict['detection_text_settings'])
            del config_dict['detection_text_settings']
        if 'detection_settings' in config_dict:
            self.detection_settings.update(config_dict['detection_settings'])
            del config_dict['detection_settings']


        self.__dict__.update(config_dict)

    @classmethod
    def from_file(cls, file_path):
        with open(file_path) as f:
            data = f.read()
        config_dict = json.loads(data)
        
        return cls(config_dict)
