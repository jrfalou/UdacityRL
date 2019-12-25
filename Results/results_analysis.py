import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

episodes_list = np.arange(100, 1001, 100)

def get_results_df(results_file, tops_nb=5):
    results_df = pd.read_csv(results_file, names=['model','episode','score'])
    solved_df = results_df[['model','episode']].groupby('model').max()
    tops_df = solved_df.sort_values(by=['episode']).iloc[:tops_nb]

    #complete missing episodes post-solved
    missing_ep_df = pd.DataFrame(columns=['model', 'episode', 'score'])
    ep_id = 0
    for model in solved_df.index:
        for ep in np.arange(100, 1001, 100):
            if (ep <= int(solved_df.loc[model])):
                continue
            missing_ep_df.loc[ep_id] = [model, ep, 14]
            ep_id += 1
    results_df = pd.concat([results_df, missing_ep_df]).sort_values(by=['model', 'episode'])
    results_df['model_type'] = results_df.apply(lambda row: row['model'][:row['model'].find('_[')], axis=1)

    return results_df, solved_df, tops_df

def plot_results(results_df, models):
    tr_dict = {}
    for model in models:
        tr_dict[model] = []
        for ep in episodes_list:
            tr_dict[model].append(float(results_df[(results_df['model']==model) & (results_df['episode']==ep)]['score'].astype('float')))
            
    tr_df = pd.DataFrame.from_dict(tr_dict)
    tr_df.set_index(episodes_list, inplace=True)

    tr_df[models].plot(kind='bar', figsize=(15,5))
    return tr_df

def plot_results_per_model_type(results_df):
    mt_df = results_df[['model_type','episode','score']].groupby(['model_type','episode']).sum()
    mt_df_count = results_df[['model_type','episode','score']].groupby(['model_type','episode']).count()
    mt_dict = {}
    for mt in results_df['model_type'].unique():
        mt_dict[mt] = []
        for ep in episodes_list:
            mt_dict[mt].append(float(mt_df.loc[(mt, ep)])/float(mt_df_count.loc[(mt, ep)]['score']))
            
    mt_df = pd.DataFrame.from_dict(mt_dict)
    mt_df.set_index(episodes_list, inplace=True)
    mt_df.plot(kind='bar', figsize=(15,5))
    return mt_df

if __name__ == "__main__":
    #get results
    results_df, solved_df, tops_df = get_results_df(result_file='train_results.csv', tops_nb=5)

    #build winners plot
    plot_results(results_df, tops_df.index)

    #build model_type plot
    plot_results_per_model_type(results_df)