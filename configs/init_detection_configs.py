# Low/medium/high detects 6/12/18 classes per picture on average

config_single_low = {
    'init_strat' : 'single',
    'init_settings' : {
        'threshold' : 8.9e-3,
    },
    'batch_size' : 64,
}
config_single_med= {
    'init_strat' : 'single',
    'init_settings' : {
        'threshold' : 3e-3,
    },
    'batch_size' : 64,
}
config_single_high = {
    'init_strat' : 'single',
    'init_settings' : {
        'threshold' : 1.45e-3,
    },
    'batch_size' : 64,
}


config_segments_low = {
    'init_strat' : 'segments',
    'init_settings' : {
        'num_segments' : 4,
        'threshold' : 2.15e-1,
    },
    'batch_size' : 64,
}
config_segments_med = {
    'init_strat' : 'segments',
    'init_settings' : {
        'num_segments' : 4,
        'threshold' : 1e-1,
    },
    'batch_size' : 64,
}
config_segments_high= {
    'init_strat' : 'segments',
    'init_settings' : {
        'num_segments' : 4,
        'threshold' : 6.1e-2,
    },
    'batch_size' : 64,
}


config_single_repeat_low = {
    'init_strat' : 'single',
    'init_settings' : {
        'repeat_wo_best' : True,
        'max_repeat_steps' : 3,
        'threshold' : 8.3e-2,
    },
    'batch_size' : 64,
}
config_single_repeat_med = {
    'init_strat' : 'single',
    'init_settings' : {
        'repeat_wo_best' : True,
        'max_repeat_steps' : 3,
        'threshold' : 4.6e-2,
    },
    'batch_size' : 64,
}
config_single_repeat_high = {
    'init_strat' : 'single',
    'init_settings' : {
        'repeat_wo_best' : True,
        'max_repeat_steps' : 3,
        'threshold' : 3.3e-2,
    },
    'batch_size' : 64,
}


config_segments_repeat_low = {
    'init_strat' : 'segments',
    'init_settings' : {
        'num_segments' : 4,
        'repeat_wo_best' : True,
        'max_repeat_steps' : 3,
        'threshold' : 4.06e-1,
    },
    'batch_size' : 64,
}
config_segments_repeat_med = {
    'init_strat' : 'segments',
    'init_settings' : {
        'num_segments' : 4,
        'repeat_wo_best' : True,
        'max_repeat_steps' : 3,
        'threshold' : 2.42e-1,
    },
    'batch_size' : 64,
}
config_segments_repeat_high = {
    'init_strat' : 'segments',
    'init_settings' : {
        'num_segments' : 4,
        'repeat_wo_best' : True,
        'max_repeat_steps' : 3,
        'threshold' : 1.83e-1,
    },
    'batch_size' : 64,
}