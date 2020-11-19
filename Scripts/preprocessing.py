import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/scratch/bay1/')
    args, _ = parser.parse_known_args()
    return args

def main():
    df_train = pd.read_csv(f'{args.data_dir}/train_subset_00.csv')

    skf = StratifiedKFold(5, shuffle=True, random_state=233)

    df_train['fold'] = -1
    for i, (_, valid_idx) in enumerate(skf.split(df_train, df_train['landmark_id'])):
        df_train.loc[valid_idx, 'fold'] = i
        
    df_train.to_csv(f'{args.data_dir}/train_0.csv', index=False)


    landmark_id2idx = {idx:landmark_id for idx, landmark_id in enumerate(sorted(df_train['landmark_id'].unique()))}
    with open('idx2landmark_id.pkl', 'wb') as fp:
        pickle.dump(landmark_id2idx, fp)

if __name__ == '__main__':

    args = parse_args() 

    main()