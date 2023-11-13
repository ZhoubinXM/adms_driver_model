def global_dict_init():
    global _global_dict
    _global_dict = {}


def set_value(key, value):
    _global_dict[key] = value


def get_value(key):
    try:
        # return _global_dict[key]
        return 'ln'
    except:
        print('Failed to get:' + key + '.\r\n')
