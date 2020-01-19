import json
import argparse
from utils import get_layers_list_from_string, get_list_from_string

def generate(n_episodes,
            actor_lr,
            critic_lr,
            actor_hidden_layers,
            critic_hidden_layers,
            noise_theta,
            noise_sigma,
            batch_sizes,
            learn_step_nbs):
    out_dict = []
    params_id = 0

    for batch_size in batch_sizes:
        for learn_step_nb in learn_step_nbs:
            batch_size = int(batch_size)
            learn_step_nb = int(learn_step_nb)
            for actor_hidden_layer in actor_hidden_layers:
                for critic_hidden_layer in critic_hidden_layers:
                    out_dict.append({})
                    out_dict[params_id]['id'] = params_id
                    out_dict[params_id]['training'] = {}
                    out_dict[params_id]['training']['n_episodes'] = int(n_episodes)
                    out_dict[params_id]['training']['max_t'] = 10000
                    out_dict[params_id]['training']['max_score'] = 30
                    # out_dict[params_id]['training']['eps_start'] = 1.
                    # out_dict[params_id]['training']['eps_end'] = 0.01
                    # out_dict[params_id]['training']['eps_decay'] = eps_decay
                    out_dict[params_id]['training']['max_consec_worse'] = 10000
                    out_dict[params_id]['training']['max_total_worse'] = 10000
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
                    out_dict[params_id]['agent']['batch_size'] = batch_size
                    out_dict[params_id]['agent']['learning_rate_actor'] = actor_lr
                    out_dict[params_id]['agent']['learning_rate_critic'] = critic_lr
                    out_dict[params_id]['agent']['learn_step_nb'] = learn_step_nb
                    out_dict[params_id]['agent']['noise_theta'] = noise_theta
                    out_dict[params_id]['agent']['noise_sigma'] = noise_sigma
                    out_dict[params_id]['agent']['actor_hidden_layers'] = str(actor_hidden_layer)
                    out_dict[params_id]['agent']['critic_hidden_layers'] = str(critic_hidden_layer)
                    out_dict[params_id]['agent']['model_tag'] = '_'.join([
                        str(batch_size),
                        str(learn_step_nb),
                        str(actor_lr),
                        str(critic_lr),
                        str(noise_theta),
                        str(noise_sigma),
                        str(actor_hidden_layer),
                        str(critic_hidden_layer)])
                    params_id += 1

    return {'params': out_dict}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters Generator for continuous control project')
    
    parser.add_argument('--name', dest='name', default='train_default')
    parser.add_argument('--batch_sizes', dest='batch_sizes', default=[20])
    parser.add_argument('--learn_step_nbs', dest='learn_step_nbs', default=[1])
    parser.add_argument('--noise_theta', dest='noise_theta', default=0)
    parser.add_argument('--noise_sigma', dest='noise_sigma', default=0)
    parser.add_argument('--actor_hidden_layers', dest='actor_hidden_layers', default=[(400, 300)])
    parser.add_argument('--critic_hidden_layers', dest='critic_hidden_layers', default=[(400, 300)])
    parser.add_argument('--nb_episodes', dest='nb_episodes', default=1000)
    parser.add_argument('--actor_lr', dest='actor_lr', default=0.0001)
    parser.add_argument('--critic_lr', dest='critic_lr', default=0.001)
    args = parser.parse_args()

    out_dict = generate(n_episodes=args.nb_episodes,
                        actor_lr=float(args.actor_lr),
                        critic_lr=float(args.critic_lr),
                        actor_hidden_layers=get_layers_list_from_string(args.actor_hidden_layers),
                        critic_hidden_layers=get_layers_list_from_string(args.critic_hidden_layers),
                        noise_theta=float(args.noise_theta),
                        noise_sigma=float(args.noise_sigma),
                        batch_sizes=get_list_from_string(args.batch_sizes),
                        learn_step_nbs=get_list_from_string(args.learn_step_nbs))

    out_dict.update({'desc': str(args)})

    out_file = open('./' + args.name + '.json', 'w')
    out_file.write(json.dumps(out_dict, indent=4, sort_keys=True))
    out_file.close()