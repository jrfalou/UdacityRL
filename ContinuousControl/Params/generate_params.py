import numpy as np
import json

out_dict = []
params_id = 0

batch_sizes = np.arange(10, 201, 10)
print('batch_sizes', batch_sizes)
for batch_size in batch_sizes:
    out_dict.append({})
    out_dict[params_id]['id'] = params_id
    out_dict[params_id]['training'] = {}
    out_dict[params_id]['training']['n_episodes'] = 300
    out_dict[params_id]['training']['max_t'] = 1000
    out_dict[params_id]['training']['max_score'] = 30
    # out_dict[params_id]['training']['eps_start'] = 1.
    # out_dict[params_id]['training']['eps_end'] = 0.01
    # out_dict[params_id]['training']['eps_decay'] = eps_decay
    out_dict[params_id]['training']['max_consec_worse'] = 1000
    out_dict[params_id]['training']['max_total_worse'] = 1000
    out_dict[params_id]['agent'] = {}
    # out_dict[params_id]['agent']['nn_layers'] = layers_params
    # out_dict[params_id]['agent']['memory_prio_params'] = {}
    # out_dict[params_id]['agent']['memory_prio_params']['memory_prio_enabled'] = False
    # out_dict[params_id]['agent']['memory_prio_params']['memory_prio_a'] = 0
    # out_dict[params_id]['agent']['memory_prio_params']['memory_prio_b0'] = 0
    # out_dict[params_id]['agent']['memory_prio_params']['memory_prio_b_step'] = 0
    # out_dict[params_id]['agent']['double_dqn'] = True if model_type == 'double' else False 
    # out_dict[params_id]['agent']['step_reward'] = -1 if model_type == 'fast' else 0
    out_dict[params_id]['agent']['debug_mode'] = False
    out_dict[params_id]['agent']['soft_tau'] = 1e-3
    out_dict[params_id]['agent']['batch_size'] = int(batch_size)
    out_dict[params_id]['agent']['learning_rate_actor'] = 1e-4
    out_dict[params_id]['agent']['learning_rate_critic'] = 1e-3
    out_dict[params_id]['agent']['model_tag'] = '_'.join([str(batch_size)])
    params_id += 1

out_dict = {'params': out_dict}

out_file = open('./training_params.json', 'w')
out_file.write(json.dumps(out_dict, indent=4, sort_keys=True))
out_file.close()