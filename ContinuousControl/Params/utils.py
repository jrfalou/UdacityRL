def get_list_from_string(str_list):
    value_list = str_list.replace(' ', '').strip('][').split(',')
    return [float(v) for v in value_list]

def get_layers_list_from_string(str_layers):
    layers_list = str_layers.replace(' ', '').strip(']()[').split('),(')
    return [(int(a.split(',')[0]), int(a.split(',')[1])) for a in layers_list]