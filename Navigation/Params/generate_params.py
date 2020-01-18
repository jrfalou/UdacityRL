import numpy as np
import json

out_dict = []
params_id = 0

layers_params_list = [[64, 64, 64], [64, 64], [32, 32, 32], [32, 32]]
epsilon_decay_list = np.arange(0.994, 0.997, 0.001)
model_types = ['simple', 'double']
for model_type in model_types:
    for layers_params in layers_params_list:
        for eps_decay in epsilon_decay_list:
            out_dict.append({})
            out_dict[params_id]['id'] = params_id
            out_dict[params_id]['training'] = {}
            out_dict[params_id]['training']['n_episodes'] = 1000
            out_dict[params_id]['training']['max_t'] = 1000
            out_dict[params_id]['training']['max_score'] = 14
            out_dict[params_id]['training']['eps_start'] = 1.
            out_dict[params_id]['training']['eps_end'] = 0.01
            out_dict[params_id]['training']['eps_decay'] = eps_decay
            out_dict[params_id]['training']['max_consec_worse'] = 10
            out_dict[params_id]['training']['max_total_worse'] = 10
            out_dict[params_id]['agent'] = {}
            out_dict[params_id]['agent']['nn_layers'] = layers_params
            out_dict[params_id]['agent']['memory_prio_params'] = {}
            out_dict[params_id]['agent']['memory_prio_params']['memory_prio_enabled'] = False
            out_dict[params_id]['agent']['memory_prio_params']['memory_prio_a'] = 0
            out_dict[params_id]['agent']['memory_prio_params']['memory_prio_b0'] = 0
            out_dict[params_id]['agent']['memory_prio_params']['memory_prio_b_step'] = 0
            out_dict[params_id]['agent']['double_dqn'] = True if model_type == 'double' else False 
            out_dict[params_id]['agent']['step_reward'] = -1 if model_type == 'fast' else 0
            out_dict[params_id]['agent']['debug_mode'] = False
            out_dict[params_id]['agent']['soft_tau'] = 1e-3
            out_dict[params_id]['agent']['learning_rate'] = 5e-4
            out_dict[params_id]['agent']['model_tag'] = '_'.join([model_type, str(layers_params), str(eps_decay)])
            params_id += 1

model_type = 'prio'
layers_params = [64, 64, 64]
eps_decay = 0.995
prio_as = np.arange(0., 1., 0.1)
prio_b0s = np.arange(0., 1., 0.1)
for prio_a in prio_as:
    for prio_b0 in prio_b0s:
        prio_a = round(prio_a, 1)
        prio_b0 = round(prio_b0, 1)
        out_dict.append({})
        out_dict[params_id]['id'] = params_id
        out_dict[params_id]['training'] = {}
        out_dict[params_id]['training']['n_episodes'] = 1000
        out_dict[params_id]['training']['max_t'] = 1000
        out_dict[params_id]['training']['max_score'] = 14
        out_dict[params_id]['training']['eps_start'] = 1.
        out_dict[params_id]['training']['eps_end'] = 0.01
        out_dict[params_id]['training']['eps_decay'] = eps_decay
        out_dict[params_id]['training']['max_consec_worse'] = 10
        out_dict[params_id]['training']['max_total_worse'] = 10
        out_dict[params_id]['agent'] = {}
        out_dict[params_id]['agent']['nn_layers'] = layers_params
        out_dict[params_id]['agent']['memory_prio_params'] = {}
        out_dict[params_id]['agent']['memory_prio_params']['memory_prio_enabled'] = True
        out_dict[params_id]['agent']['memory_prio_params']['memory_prio_a'] = prio_a
        out_dict[params_id]['agent']['memory_prio_params']['memory_prio_b0'] = prio_b0
        out_dict[params_id]['agent']['memory_prio_params']['memory_prio_b_step'] = 0
        out_dict[params_id]['agent']['double_dqn'] = False
        out_dict[params_id]['agent']['step_reward'] = 0
        out_dict[params_id]['agent']['debug_mode'] = False
        out_dict[params_id]['agent']['soft_tau'] = 1e-3
        out_dict[params_id]['agent']['learning_rate'] = 5e-4
        out_dict[params_id]['agent']['model_tag'] = '_'.join([model_type, str(layers_params), str(eps_decay), str(prio_a), str(prio_b0)])
        params_id += 1

out_dict = {'params': out_dict}

out_file = open('./training_params.json', 'w')
out_file.write(json.dumps(out_dict))
out_file.close()