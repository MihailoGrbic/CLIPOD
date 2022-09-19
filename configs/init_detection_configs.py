# Low/medium/high detects 6/12/18 classes per picture on average

base_config = {
    'batch_size' : 64,
    'text_prompt_prepend' : "A photo of a ",
    'crop' : True,
}

init_detection_configs = {
    
    'single_low' : {
        'init_strat' : 'single',
        'init_settings' : {
            'threshold' : 8.9e-3,
        },
    },
    'single_med' : {
        'init_strat' : 'single',
        'init_settings' : {
            'threshold' : 2.96e-3,
        },
    },
    'single_high' : {
        'init_strat' : 'single',
        'init_settings' : {
            'threshold' : 1.45e-3,
        },
    },

    #====================================================================

    '3_segments_low' : {
        'init_strat' : 'segments',
        'init_settings' : {
            'num_segments' : 3,
            'threshold' : 1.37e-1,
        },
    },
    '3_segments_med' : {
        'init_strat' : 'segments',
        'init_settings' : {
            'num_segments' : 3,
            'threshold' : 6e-2,
        },
    },
    '3_segments_high': {
        'init_strat' : 'segments',
        'init_settings' : {
            'num_segments' : 3,
            'threshold' : 3.68e-2,
        },
    },

    #====================================================================

    '4_segments_low' : {
        'init_strat' : 'segments',
        'init_settings' : {
            'num_segments' : 4,
            'threshold' : 2.14e-1,
        },
    },
    '4_segments_med' : {
        'init_strat' : 'segments',
        'init_settings' : {
            'num_segments' : 4,
            'threshold' : 9.94e-2,
        },
    },
    '4_segments_high': {
        'init_strat' : 'segments',
        'init_settings' : {
            'num_segments' : 4,
            'threshold' : 6.18e-2,
        },
    },

    #====================================================================

    '5_segments_low' : {
        'init_strat' : 'segments',
        'init_settings' : {
            'num_segments' : 5,
            'threshold' : 2.85e-1,
        },
    },
    '5_segments_med' : {
        'init_strat' : 'segments',
        'init_settings' : {
            'num_segments' : 5,
            'threshold' : 1.41e-1,
        },
    },
    '5_segments_high': {
        'init_strat' : 'segments',
        'init_settings' : {
            'num_segments' : 5,
            'threshold' : 8.88e-2,
        },
    },

    #====================================================================
    #====================================================================
    #====================================================================


    'single_repeat_low' : {
        'init_strat' : 'single',
        'init_settings' : {
            'repeat_wo_best' : True,
            'max_repeat_steps' : 3,
            'threshold' : 8.55e-2,
        },
    },
    'single_repeat_med' : {
        'init_strat' : 'single',
        'init_settings' : {
            'repeat_wo_best' : True,
            'max_repeat_steps' : 3,
            'threshold' : 4.6e-2,
        },
    },
    'single_repeat_high' : {
        'init_strat' : 'single',
        'init_settings' : {
            'repeat_wo_best' : True,
            'max_repeat_steps' : 3,
            'threshold' : 3.3e-2,
        },
    },

    #====================================================================

    '3_segments_repeat_low' : {
        'init_strat' : 'segments',
        'init_settings' : {
            'num_segments' : 3,
            'repeat_wo_best' : True,
            'max_repeat_steps' : 3,
            'threshold' : 3.08e-1,
        },
    },
    '3_segments_repeat_med' : {
        'init_strat' : 'segments',
        'init_settings' : {
            'num_segments' : 3,
            'repeat_wo_best' : True,
            'max_repeat_steps' : 3,
            'threshold' : 1.8e-1,
        },
        
    },
    '3_segments_repeat_high' : {
        'init_strat' : 'segments',
        'init_settings' : {
            'num_segments' : 3,
            'repeat_wo_best' : True,
            'max_repeat_steps' : 3,
            'threshold' : 1.32e-1,
        },
    },

    #====================================================================

    '4_segments_repeat_low' : {
        'init_strat' : 'segments',
        'init_settings' : {
            'num_segments' : 4,
            'repeat_wo_best' : True,
            'max_repeat_steps' : 3,
            'threshold' : 3.95e-1,
        },
    },
    '4_segments_repeat_med' : {
        'init_strat' : 'segments',
        'init_settings' : {
            'num_segments' : 4,
            'repeat_wo_best' : True,
            'max_repeat_steps' : 3,
            'threshold' : 2.42e-1,
        },
    },
    '4_segments_repeat_high' : {
        'init_strat' : 'segments',
        'init_settings' : {
            'num_segments' : 4,
            'repeat_wo_best' : True,
            'max_repeat_steps' : 3,
            'threshold' : 1.805e-1,
        },
    },

    #====================================================================

    '5_segments_repeat_low' : {
        'init_strat' : 'segments',
        'init_settings' : {
            'num_segments' : 5,
            'repeat_wo_best' : True,
            'max_repeat_steps' : 3,
            'threshold' : 4.63e-1,
        },
    },
    '5_segments_repeat_med' : {
        'init_strat' : 'segments',
        'init_settings' : {
            'num_segments' : 5,
            'repeat_wo_best' : True,
            'max_repeat_steps' : 3,
            'threshold' : 2.98e-1,
        },
    },
    '5_segments_repeat_high' : {
        'init_strat' : 'segments',
        'init_settings' : {
            'num_segments' : 5,
            'repeat_wo_best' : True,
            'max_repeat_steps' : 3,
            'threshold' : 2.265e-1,
        },
    },

}