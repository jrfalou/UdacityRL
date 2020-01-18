#external imports
import argparse
import json
import os
from unityagents import UnityEnvironment

#local imports
from agent_trainer import AgentTrainer
from time_analysis import TimeAnalysis

def create_results_folders(filepath):
    path_parts = filepath.split('/')
    if len(path_parts) > 1:
        for p in range(len(path_parts)-1):
            full_path = './Results/' + '/'.join(path_parts[:-(p+1)])
            print(full_path)
            if not os.path.exists(full_path):
                os.mkdir(full_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters for continuous control project')
    parser.add_argument('mode')
    
    parser.add_argument('--test_model', dest='test_model', default='random')
    parser.add_argument('--test_params', dest='test_params', default='default_params.json')
    parser.add_argument('--test_no_display', dest='test_no_display', action='store_true')
    parser.add_argument('--test_results_path', dest='test_results_path', default='test_results.csv')
    parser.add_argument('--test_episodes', dest='test_episodes', default=1)
    
    parser.add_argument('--train_params', dest='train_params', default='default_params.json')
    parser.add_argument('--train_start_id', dest='train_start_id', default=0)
    parser.add_argument('--train_results_path', dest='train_results_path', default='train_results.csv')
    parser.add_argument('--train_debug', dest='train_debug', default=False)
    parser.add_argument('--train_worker_id', dest='train_worker_id', default=0)
    args = parser.parse_args()
    
    env = None
    try:
        if args.mode == 'test':
            params_file = open('./Params/' + args.test_params, 'r')
            test_params = json.loads(params_file.read())
            params_file.close()
            
            env = UnityEnvironment(
                file_name='./Reacher_Linux_1/Reacher_Linux/Reacher.x86_64', #"./Reacher_Windows_x86_64/Reacher.exe",
                no_graphics=args.test_no_display
            )
            for testing_params in test_params['params']:
                for i in range(int(args.test_episodes)):
                    agent_trainer = AgentTrainer(
                        env=env, 
                        params=testing_params,
                        results_path='./Results/' + args.test_results_path if args.test_results_path != '' else ''
                    )
                    test_model = args.test_model
                    if test_model == 'auto':
                        test_model = testing_params['agent']['model_tag']
                    agent_trainer.test(model_weights=test_model, test_id=i+1)
        elif args.mode == 'train':
            params_file = open('./Params/' + args.train_params, 'r')
            train_params = json.loads(params_file.read())
            params_file.close()

            create_results_folders(args.train_results_path)

            env = UnityEnvironment(
                file_name='./Reacher_Linux_1/Reacher_Linux/Reacher.x86_64', #"./Reacher_Windows_x86_64/Reacher.exe",
                no_graphics=True,
                worker_id=int(args.train_worker_id)
            )
            for training_params in train_params['params']:
                if(training_params['id'] < int(args.train_start_id)):
                    continue

                time_analysis = TimeAnalysis()
                agent_trainer = AgentTrainer(
                    env=env,
                    params=training_params,
                    results_path='./Results/' + args.train_results_path,
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
