import torch, pandas as pd, numpy as np, sys, os, re
from matplotlib import pyplot as plt
from tabulate import tabulate
sys.path.append('/'.join(re.split('/|\\\\', os.path.dirname( __file__ ))[0:-1]))
from derived.simulation import *
from rddesign.main import *

def main():
    outdir = 'output/tables'
    ns = [500, 2000, 8000]
    models = {'0': model_0, '1': model_1, '2': model_2, '3': model_3}
    TEs = {'0': 1, '1': 0.550595 - 0.443452, '2': 0.375062 - 3.590380, '3': 0}
    DGPs = {'no_confounding': sim_unbiased, 'confounding': sim_biased}
    band_nsims = 10
    nsims = 400 - band_nsims
    
    for dgp in DGPs.keys():
        rows = []
        for n in ns:
            row = pd.DataFrame({'$\\tilde{\\mu}_{j}(\\cdot) = $': f'n = {n}', 'bias': '', 'Coverage': '', 'Length$': '',  '$h$': '',
                                'bias\\vphantom{l}': '', 'Coverage\\vphantom{l}': '', 'Length\\vphantom{l}': '', '$h$\\vphantom{l}': ''}, index = [0])
            rows.append(row.set_index('$\\tilde{\\mu}_{j}(\\cdot) = $'))
            for model in models.keys():
                band_pos_pdd, band_neg_pdd = [], []
                band_pos_rdd, band_neg_rdd = [], []
                reps, successes = 10042002, 0
                print(n, 'Model %s' % model)
                '''
                while successes < band_nsims:
                    Y, W, D, Z, U = DGPs[dgp](models[model], ndraws = n, seed = reps)
                    design = pdd(Y, W, D, Z, cutoff = 0.0, device = 'cuda', kernel = 'triangle')
                    res_pdd = design.fit()
                    
                    design = rdd(Y, D, cutoff = 0.0, device = 'cuda', kernel = 'triangle')
                    res_rdd = design.fit()
                    
                    if res_pdd.status == True and res_rdd.status == True:
                        reps += 1
                        successes += 1
                        band_pos_pdd.append(res_pdd.bandwidth['+'])
                        band_neg_pdd.append(res_pdd.bandwidth['-'])
                        band_pos_rdd.append(res_rdd.bandwidth['+'])
                        band_neg_rdd.append(res_rdd.bandwidth['-'])
                    else:
                        reps += 1
                    print(successes, reps)
                '''
                band_pdd = [0.2 * n**(-1/5)/500**(-1/5), 0.2 * n**(-1/5)/500**(-1/5)] #[np.mean(band_neg_pdd), np.mean(band_pos_pdd)]
                band_rdd = [0.2 * n**(-1/5)/500**(-1/5), 0.2 * n**(-1/5)/500**(-1/5)] #[np.mean(band_neg_rdd), np.mean(band_pos_rdd)]
                covered_pdd, length_pdd, se_pdd, est_pdd = [], [], [], []
                covered_rdd, length_rdd, se_rdd, est_rdd = [], [], [], []

                while successes < band_nsims + nsims:
                    Y, W, D, Z, U = DGPs[dgp](models[model], ndraws = n, seed = reps)
                    design = pdd(Y, W, D, Z, cutoff = 0.0, device = 'cuda', kernel = 'triangle', bandwidth = band_pdd)
                    res_pdd = design.fit()
                    
                    design = rdd(Y, D, cutoff = 0.0, device = 'cuda', kernel = 'triangle', bandwidth = band_rdd)
                    res_rdd = design.fit()
                    if res_pdd.status == True and res_rdd.status == True:
                        reps += 1
                        successes += 1
                        covered_pdd.append(1 if TEs[model] >= res_pdd.left_ci and TEs[model] <= res_pdd.right_ci else 0)
                        covered_rdd.append(1 if TEs[model] >= res_rdd.left_ci and TEs[model] <= res_rdd.right_ci else 0)
                        length_pdd.append(res_pdd.right_ci - res_pdd.left_ci)
                        length_rdd.append(res_rdd.right_ci - res_rdd.left_ci)
                        se_pdd.append(res_pdd.se)
                        se_rdd.append(res_rdd.se)
                        est_pdd.append(res_pdd.est)
                        est_rdd.append(res_rdd.est)
                    else:
                        reps += 1

                row = pd.DataFrame({'$\\tilde{\\mu}_{j}(\\cdot) = $': '$\\tilde{\\mu}_{%s}(\\cdot)$\\vphantom{%d}' % (model, n), 'bias': np.mean(est_pdd) - TEs[model], 
                                    'Coverage': 100 * np.mean(covered_pdd), 'Length': np.mean(length_pdd),  '$h$': band_pdd[0],
                                    'bias\\vphantom{l}': np.mean(est_rdd) - TEs[model], 'Coverage\\vphantom{l}': 100 * np.mean(covered_rdd), 'Length\\vphantom{l}$': np.mean(length_rdd), 
                                    '$h$\\vphantom{l}': band_rdd[0]}, index = [0])
                rows.append(row.set_index('$\\tilde{\\mu}_{j}(\\cdot) = $'))
            
        df_res = pd.concat(rows, axis = 0)
        df_res = df_res.apply(pd.to_numeric, errors="coerce").astype("float64")
        df_res = df_res.round(2)
        table = '\\renewcommand{\\arraystretch}{1.5}\n' + '\\begin{table}[H]\n' + \
                    '\\begin{center}\n' + '<insert caption>\n' + \
                    tabulate(df_res, headers = 'keys', tablefmt="latex_raw") + \
                    '\n\\end{center}\n' + '\\medskip\n' + '\\end{table}\n'
        table = table.replace('\\begin{tabular}{lrrrrrrrrrr}', '\\begin{tabular}{l@{\\hskip 1em}c@{\\hskip 1em}c@{\\hskip 1em}c@{\\hskip 1em}c@{\\hskip 1em}' +\
                                'c@{\\hskip 1em}c@{\\hskip 1em}c@{\\hskip 1em}c@{\\hskip 1em}c@{\\hskip 1em}c}\n\\toprule\n' +\
                                '& \\multicolumn{4}{c}{PDD} & \\multicolumn{4}{c}{RDD} \\\\[-0.25em] \n' +\
                                '\\cmidrule(r{2em}){2-5} \\cmidrule(r){6-9} & \\\\[-1.75em]')
        table = table.replace('nan', '')
        file = open(f'{outdir}/{dgp}4k.tex', 'w')
        file.write(table)
        file.close()
                

if __name__ == '__main__':
    main()
