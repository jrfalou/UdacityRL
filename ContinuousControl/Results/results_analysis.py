import glob
import pandas as pd

MODEL_TAGS = ['batch_size', 
            'learn_step', 
            'actor_lr', 
            'critic_lr', 
            'noise_theta', 
            'noise_sigma', 
            'actor_nn', 
            'critic_nn']

def get_training_results():
    results_dirs = glob.glob('./training_*/')
    df_results_step_list = []
    df_results_list = []
    df_results_details_list = []
    for results_dir in results_dirs:
        df_step = pd.read_csv(
            results_dir + 'train_results_step.csv',
            names=['model_id', 'model_tag', 'episode', 'score'])
        df = pd.read_csv(
            results_dir + 'train_results.csv',
            names=['model_id', 'model_tag', 'episode', 'score'])
        df_step['training'] = results_dir.split('/')[-2]
        df['training'] = results_dir.split('/')[-2]
        df_results_step_list.append(df_step)
        df_results_list.append(df)
        
        training_batch = results_dir.replace('/', '').replace('.', '')
        df_details = df.copy()
        df_details['batch'] = training_batch
        for i in range(len(MODEL_TAGS)):
            model_tag = MODEL_TAGS[i]
            df_details[model_tag] = df_details.apply(
                lambda row: row['model_tag'].split('_')[i],
                axis=1)
        df_results_details_list.append(df_details)

    df_results_step = pd.concat(df_results_step_list, axis=0, sort=False)
    df_results = pd.concat(df_results_list, axis=0, sort=False)
    df_details = pd.concat(df_results_details_list, axis=0, sort=False)

    return df_results_step, df_results, df_details

def get_best_runs(df_results_step, df_results, nb_runs=5):
    df_results_step_max = df_results_step[['model_tag', 'score', 'training']].\
        groupby(['training', 'model_tag']).max()
    df_results_step_max = df_results_step_max.sort_values('score', ascending=False)

    df_results_max = df_results[['model_tag','score', 'training']].\
        groupby(['training', 'model_tag']).max()
    df_results_max = df_results_max.sort_values('score', ascending=False)

    return df_results_step_max, df_results_max

def plot_model(df_results_step, df_results, training_run, model_tag, by_step=True, kind='line'):
    if by_step:
        df = df_results_step.copy()
    else:
        df = df_results.copy()
    df_plot = df[(df['model_tag'] == model_tag) & (df['training'] == training_run)][['episode', 'score']]
    df_plot = df_plot.set_index('episode')
    df_plot.plot(kind=kind, title=(model_tag + '(' + training_run +')'), figsize=(15, 5))

def get_stats_per_parameter(df_details):
    df_details_summary = df_details[df_details['episode'] == 1000]
    model_tags_dict = {}
    for model_tag in MODEL_TAGS:
        model_tags_dict[model_tag] = df_details_summary[model_tag].unique()

    summary_dict = {'max': [], 'mean': [], 'count': []}
    index_list = []
    for t in model_tags_dict:
        df_max = df_details_summary[[t, 'score']].groupby(t).max()
        df_mean = df_details_summary[[t, 'score']].groupby(t).mean()
        df_count = df_details_summary[[t, 'score']].groupby(t).count()
        for v in model_tags_dict[t]:
            summary_dict['max'].append(round(df_max.loc[v]['score'], 2))
            summary_dict['mean'].append(round(df_mean.loc[v]['score'], 2))
            summary_dict['count'].append(df_count.loc[v]['score'])
            index_list.append((t, v))

    summary_df = pd.DataFrame.from_dict(summary_dict)
    index_mult = pd.MultiIndex.from_tuples(index_list, names=['tag_type', 'tag_value'])
    summary_df = summary_df.set_index(index_mult)
    return summary_df