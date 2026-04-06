import numpy as np, pandas as pd, warnings

def main():
    indir = 'src/raw'
    outdir = 'output/derived/almond'
    df = clean_data(indir, ['linkco2014usden', 'linkco2015usden'])
    df_out = transform_data(df)
    df_out.to_csv(f'{outdir}/clean_data.csv', index = False)

def transform_data(df):
    
    aged = df.loc[:, 'aged'].values
    death = np.where(aged != 1, 1, 0)
    educ = df.loc[:, 'meduc'].values
    is_educ = np.where(educ >= 5, 1, 0)
    pay = df.loc[:, 'pay'].values
    medicaid = np.where(pay == 1, 1, 0)
    tob = df.loc[:, 'dob_tt'].values
    night = np.where((tob < 700) + (tob > 1800), 1, 0)
    dob = df.loc[:, 'dob_wk'].values
    wknd = np.where((dob == 1) + (dob == 7), 1, 0)
    
    df_out = pd.DataFrame({'death': death, 'meduc': is_educ, 'medicaid': medicaid, 
                           'night': night, 'wknd': wknd, 'brthwgt': df.loc[:, 'brthwgt'].values}, 
                          index = df.index)
    return df_out
    
def clean_data(indir, files):
    dfs = []
    for file in files:
        df_raw = pd.read_csv(f'{indir}/{file}.csv')
        data_vars = ['aged', 'brthwgt', 'meduc', 'pay', 'dob_tt', 'dob_wk']
        
        df = df_raw.loc[:, data_vars]
        df = df.dropna(axis = 0).reset_index(drop=True)
        
        wgt = df.loc[:, 'brthwgt'].values
        close_wgt = np.where(np.abs(wgt - 1500) <= 50, True, False)
        df = df.loc[close_wgt, :].reset_index(drop=True)
        
        meduc = df.loc[:, 'meduc'].values
        known_meduc = np.where(meduc != 9, True, False)
        df = df.loc[known_meduc, :].reset_index(drop=True)
        
        paytype = df.loc[:, 'pay'].values
        known_pay = np.where(paytype != 9, True, False)
        df = df.loc[known_pay, :].reset_index(drop=True)
        
        tob = df.loc[:, 'dob_tt'].values
        known_tob = np.where(tob < 9000, True, False)
        df = df.loc[known_tob, :].reset_index(drop=True)

        dfs.append(df)
    
    df = pd.concat(dfs, axis = 0)
    return df

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
