base_config = {
    'batch_size' : 128,
    'text_prompt_prepend' : "A photo of a ",
    'crop' : True,
 
    'init_strat' : 'segments',
    'init_settings' : {
        'num_segments' : 4,
        'repeat_wo_best' : True,
        'max_repeat_steps' : 3,
        'threshold' : 2.42e-1,
    }
}

text_strat_configs = {
    'negative_text' : 
    {
        'detection_text_strat' : 'negative-text',
        'detection_text_settings' : {
            'negative_text' : ['background', 'a photo of a background', 'clutter', 'a photo of clutter', 
                                'other', 'random', 'stuff', 'random stuff', 'a photo of random stuff',
                                'junk', 'a photo of junk', 'random junk', 'a photo of random junk',
                                'backdrop', 'nothing', 'a photo of nothing', 'mess', 'a photo of a mess',
                                'noise', 'a photo of noise', 'gaussian noise', 'a photo of gaussian noise'],
        },
    },
    'negative_cats' : 
    {
        'detection_text_strat' : 'negative-cats',
        'detection_text_settings' : {
            'num_other_cats' : 100,
        }
    },
    'promising_cats' : 
    {
        'detection_text_strat' : 'promising-cats',
        'detection_text_settings' : {
            'num_other_cats' : 100,
        }
    },
    'all_cats' : 
    {
        'detection_text_strat' : 'all-cats',
    }
}

detection_strat_configs = {
    'random' : 
    {
        'detection_strat' : 'random',
        'detection_settings' : {
            'num_tries' : 100,
        }
    },
    'beam_search_3_3' : 
    {
        'detection_strat' : 'beam-search',
        'detection_settings' : {
            'num_segments' : 3,
            'start_threshold' : 0.05,
            'num_new' : 3,
            'num_saved' : 3,
            'new_range' : 30,
            'max_steps' : 200,
            'patience' : 10
        }
    },
    'beam_search_5_5' : 
    {
        'detection_strat' : 'beam-search',
        'detection_settings' : {
            'num_segments' : 3,
            'start_threshold' : 0.05,
            'num_new' : 5,
            'num_saved' : 5,
            'new_range' : 30,
            'max_steps' : 200,
            'patience' : 10
        }
    },
    'beam_search_10_3' : 
    {
        'detection_strat' : 'beam-search',
        'detection_settings' : {
            'num_segments' : 3,
            'start_threshold' : 0.05,
            'num_new' : 3,
            'num_saved' : 10,
            'new_range' : 30,
            'max_steps' : 200,
            'patience' : 10
        }
    },
    'beam_search_10_10' : 
    {
        'detection_strat' : 'beam-search',
        'detection_settings' : {
            'num_segments' : 3,
            'start_threshold' : 0.05,
            'num_new' : 3,
            'num_saved' : 10,
            'new_range' : 30,
            'max_steps' : 200,
            'patience' : 10
        }
    },
    'beam_search_single_seg' : 
    {
        'detection_strat' : 'beam-search',
        'detection_settings' : {
            'num_segments' : 1,
            'start_threshold' : 0,
            'num_new' : 5,
            'num_saved' : 5,
            'new_range' : 30,
            'max_steps' : 200,
            'patience' : 10
        }
    }
}
