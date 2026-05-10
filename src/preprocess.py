import pandas as pd
import numpy as np

def load_data(data_path, poi_cat_path):
    """
    Load and preprocess yjmob100k dataset.
    
    Args:
        data_path: path to filtered_clean.parquet
        poi_cat_path: path to cell_POIcat.csv
    
    Returns:
        df: movement data with POI category
        user_sequences: per-user POI sequences
    """
    # 데이터 로드
    df = pd.read_parquet(data_path)
    poi_cat = pd.read_csv(poi_cat_path)

    # day 27 NA 제거
    df = df.dropna(subset=['x', 'y'])

    # dtype 최적화
    df['uid'] = df['uid'].astype('int32')
    df['d'] = df['d'].astype('int8')
    df['t'] = df['t'].astype('int8')
    df['x'] = df['x'].astype('int16')
    df['y'] = df['y'].astype('int16')

    # 셀별 대표 POI 카테고리 추출
    poi_dominant = poi_cat.groupby(['x', 'y']).apply(
        lambda g: g.loc[g['POI_count'].idxmax(), 'POIcategory'],
        include_groups=False
    ).reset_index()
    poi_dominant.columns = ['x', 'y', 'poi_cat']

    # POI 카테고리 조인
    df = df.merge(poi_dominant, on=['x', 'y'], how='left')
    df['poi_cat'] = df['poi_cat'].fillna(0).astype('int8')

    return df

def make_sequences(df):
    """
    Create per-user POI sequences sorted by time.
    """
    df = df.sort_values(['uid', 'd', 't'])
    user_sequences = df.groupby('uid')['poi_cat'].apply(list).reset_index()
    user_sequences.columns = ['uid', 'poi_sequence']
    user_sequences['seq_len'] = user_sequences['poi_sequence'].apply(len)
    return user_sequences

def make_sliding_window(user_sequences, window_size=10):
    """
    Create sliding window samples from user sequences.
    
    Args:
        user_sequences: DataFrame with uid and poi_sequence
        window_size: number of past POIs to use as input
    
    Returns:
        X: input sequences (n_samples, window_size)
        y: target POI (n_samples,)
        uids: user id for each sample (n_samples,)
    """
    X, y, uids = [], [], []

    for _, row in user_sequences.iterrows():
        seq = row['poi_sequence']
        uid = row['uid']
        for i in range(len(seq) - window_size):
            X.append(seq[i:i+window_size])
            y.append(seq[i+window_size])
            uids.append(uid)

    X = np.array(X, dtype=np.int8)
    y = np.array(y, dtype=np.int8)
    uids = np.array(uids, dtype=np.int32)

    return X, y, uids
