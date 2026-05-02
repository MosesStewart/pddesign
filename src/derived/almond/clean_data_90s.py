import numpy as np, pandas as pd, warnings

def main():
    indir = 'src/raw'
    outdir = 'output/derived/almond'
    df = clean_data(indir, ['linkco%sus_den' % year for year in range(1999, 2003)])
    df_out = transform_data(df)
    df_out.to_csv(f'{outdir}/clean_data_90s.csv', index = False)

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    aged = df.loc[:, 'aged'].values
    death = np.where(aged <= 365, 1, 0)
    educ = df.loc[:, 'dmeduc'].values
    is_educ = np.where(educ >= 12, 1, 0)
    mrace = df.loc[:, 'mrace'].values
    is_white = np.where(mrace == 1, 1, 0)
    dob = df.loc[:, 'weekdayb'].values
    wknd = np.where((dob == 1) + (dob == 7), 1, 0)
    
    df_out = pd.DataFrame({'death': death, 'meduc': is_educ, 'mrace': is_white, 'wknd': wknd, 'brthwgt': df.loc[:, 'dbirwt'].values}, 
                          index = df.index)
    return df_out
    
def clean_data(indir: str, files: list) -> pd.DataFrame:
    dfs = []
    for file in files:
        df_raw = pd.read_csv(f'{indir}/{file}.csv')
        data_vars = ['aged', 'dbirwt', 'dmeduc', 'mrace', 'weekdayb']
        df_raw.loc[:, 'aged'] = df_raw.loc[:, 'aged'].fillna(999)
        
        df = df_raw.loc[:, data_vars]
        df = df.dropna(axis = 0).reset_index(drop=True)
        
        wgt = df.loc[:, 'dbirwt'].values
        close_wgt = np.where(np.abs(wgt - 1500) <= 85, True, False)
        df = df.loc[close_wgt, :].reset_index(drop=True)

        dfs.append(df)
    
    df = pd.concat(dfs, axis = 0)
    return df

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
