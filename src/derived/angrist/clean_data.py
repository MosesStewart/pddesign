import numpy as np, pandas as pd

def main():
    indir = 'src/raw'
    outdir = 'output/derived/angrist'

    df5 = clean_data(indir, 'final5')
    df4 = clean_data(indir, 'final4')
    df_out = transform_data(df5, df4)
    df_out.to_csv(f'{outdir}/clean_data.csv', index=False)

def transform_data(df5: pd.DataFrame, df4: pd.DataFrame) -> pd.DataFrame:
    # -- Maimonides' Rule instrument ------------------------------------------
    # fsc = es / (floor((es - 1) / 40) + 1), where es is September enrollment.
    # floor((c_size-1)/40) gives the number of additional classes beyond the
    # first that Maimonides' rule requires: 0 for enrollment 1-40 (1 class),
    # 1 for 41-80 (2 classes), 2 for 81-120 (3 classes), etc.
    # This is func1 from Angrist's .do files, used as the IV in Tables 3-5.
    # NOTE: c_size (September enrollment) is distinct from cohsize (the spring
    # sum of actual class sizes). The paper uses September enrollment for the
    # instrument so that it is predetermined and not affected by mid-year
    # sorting (p.542).
    enrollment = df5.loc[:, 'c_size'].values
    func1 = enrollment / (np.floor((enrollment - 1) / 40) + 1)

    # -- School-level 4th-grade average scores --------------------------------
    # Weighted average across 4th-grade classes within each school, with the
    # number of test-takers as weights. Left-joined onto the 5th-grade classes.
    avg4_verb = _weighted_school_avg(df4, score_col='avgverb', weight_col='verbsize')
    avg4_math = _weighted_school_avg(df4, score_col='avgmath', weight_col='mathsize')
    school4 = pd.DataFrame({'avg4_verb': avg4_verb, 'avg4_math': avg4_math})
    df_merged = df5.join(school4, on='schlcode')

    df_out = pd.DataFrame({
        'schlcode':  df5.loc[:, 'schlcode'].values,   # school identifier
        'classid':   df5.loc[:, 'classid'].values,    # class identifier within school
        'c_size':    enrollment,                      # (1) September 5th-grade enrollment
        'classize':  df5.loc[:, 'classize'].values,   # actual spring class size (endogenous)
        'avgverb':   df5.loc[:, 'avgverb'].values,    # (2) class-avg verbal score
        'avgmath':   df5.loc[:, 'avgmath'].values,    # (2) class-avg math score
        'instrument':     func1,                           # (3) Maimonides' Rule instrument fsc
        'avg4_verb': df_merged.loc[:, 'avg4_verb'].values,  # (4) school avg 4th-grade verbal
        'avg4_math': df_merged.loc[:, 'avg4_math'].values,  # (4) school avg 4th-grade math
    }, index=df5.index)

    df_out = df_out.dropna(axis = 0).reset_index(drop=True)
    #df_out = df_out.loc[df_out.index.repeat(df_out['classize'].astype(int))].reset_index(drop=True)
    return df_out


def clean_data(indir: str, file: str) -> pd.DataFrame:
    df_raw = pd.read_stata(f'{indir}/{file}.dta', convert_categoricals=False)
    data_vars = ['schlcode', 'classid', 'c_size', 'cohsize', 'classize',
                 'c_leom', 'c_pik', 'tipuach', 'verbsize', 'avgverb',
                 'mathsize', 'avgmath']

    df = df_raw.loc[:, data_vars]

    # Fix scores that were recorded above 100 due to a data entry convention
    avgverb = df.loc[:, 'avgverb'].values
    df.loc[:, 'avgverb'] = np.where(avgverb > 100, avgverb - 100, avgverb)
    avgmath = df.loc[:, 'avgmath'].values
    df.loc[:, 'avgmath'] = np.where(avgmath > 100, avgmath - 100, avgmath)

    # Zero test-takers means the score is undefined
    verbsize = df.loc[:, 'verbsize'].values
    df.loc[verbsize == 0, 'avgverb'] = np.nan
    mathsize = df.loc[:, 'mathsize'].values
    df.loc[mathsize == 0, 'avgmath'] = np.nan

    # Sample restrictions (mirror Angrist's .do files exactly)
    classize = df.loc[:, 'classize'].values
    df = df.loc[(classize > 1) & (classize < 45)].reset_index(drop=True)

    c_size = df.loc[:, 'c_size'].values
    df = df.loc[c_size > 5].reset_index(drop=True)

    c_leom = df.loc[:, 'c_leom'].values
    c_pik  = df.loc[:, 'c_pik'].values
    df = df.loc[(c_leom == 1) & (c_pik < 3)].reset_index(drop=True)

    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis = 0).reset_index(drop=True)
    
    return df


def _weighted_school_avg(df: pd.DataFrame,
                         score_col: str,
                         weight_col: str) -> pd.Series:
    def _wavg(g):
        valid = g.dropna(subset=[score_col])
        total_w = valid[weight_col].sum()
        if total_w == 0 or len(valid) == 0:
            return np.nan
        return np.average(valid[score_col], weights=valid[weight_col])
    return df.groupby('schlcode')[[score_col, weight_col]].apply(_wavg)


if __name__ == '__main__':
    main()
