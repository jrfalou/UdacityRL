#external imports
from unityagents import UnityEnvironment
import numpy as np
# import matplotlib.pyplot as plt
import argparse
import json

#local imports
from agent_trainer import AgentTrainer
from time_analysis import TimeAnalysis

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters for navigation project')
    parser.add_argument('mode')
    parser.add_argument('--test_model', dest='test_model', default='random')
    parser.add_argument('--test_params', dest='test_params', default='./default_params.json')
    parser.add_argument('--train_params', dest='train_params', default='./default_params.json')
    parser.add_argument('--train_start_id', dest='train_start_id', default=0)
    parser.add_argument('--train_results_path', dest='train_results_path', default='./results.csv')
    parser.add_argument('--train_debug', dest='train_debug', default=False)
    args = parser.parse_args()
    
    env = None
    try:
        if args.mode == 'test':
            env = UnityEnvironment(file_name="./Banana_Windows_x86_64/Banana.exe")
            agent_trainer = AgentTrainer(env=env, params=args.test_params[0])
            agent_trainer.test(model_weights=args.test_model)
            env.close()
        elif args.mode == 'train':
            params_file = open(args.train_params, 'r')
            training_params = json.loads(params_file.read())
            params_file.close()

            env = UnityEnvironment(
                file_name="./Banana_Windows_x86_64/Banana.exe",
                no_graphics=True
            )
            for training_params in training_params['training_params']:
                if(training_params['id'] < int(args.train_start_id)):
                    continue

                time_analysis = TimeAnalysis()
                agent_trainer = AgentTrainer(
                    env=env, 
                    params=training_params, 
                    results_path=args.train_results_path,
                    debug_mode=args.train_debug,
                    time_analysis=time_analysis
                )
                agent_trainer.train()
                print(time_analysis.to_str())
            print('Training ended')
        else:
            print('Unknown mode', args.mode)
    except Exception as e:
        print('Unexpected error:', str(e))
        raise(e)
    finally:
        if env is not None:
            env.close()
