import torch, pandas as pd, numpy as np, sys, os, re
from matplotlib import pyplot as plt
from tabulate import tabulate
sys.path.append('/'.join(re.split('/|\\\\', os.path.dirname( __file__ ))[0:-1]))
from derived.simulation import *
from rddesign.main import *

def main():
    outdir = 'output/tables'
    ns = [500]#, 2500, 10000]
    models = {'0': model_0, '1': model_1}#, '2': model_2, '3': model_3}
    band_nsims = 2
    nsims = 3
    rows = []
    
    for n in ns:
        row = pd.DataFrame({'$\\tilde{\\mu}_{j}(\\cdot) = $': f'n = {n}', 'Coverage': '', '$\\overline{\\text{Length}}$': '', '$\\bar{v}_{\\text{rbc}}$': '', 
                            '$h_{+}$': '', '$h_{-}$': '', 'Coverage\\vphantom{l}': '', '$\\overline{\\text{Length\\vphantom{l}}}$': '', 
                            '$\\bar{v}_{\\text{rbc}}$\\vphantom{l}': '', '$h_{+}$\\vphantom{l}': '', '$h_{-}$\\vphantom{l}': ''}, index = [0])
        rows.append(row.set_index('$\\tilde{\\mu}_{j}(\\cdot) = $'))
        for model in models.keys():
            band_pos_pdd, band_neg_pdd = [], []
            band_pos_rdd, band_neg_rdd = [], []
            reps, successes = 0, 0
            while successes < band_nsims:
                Y, W, D, Z, U = sim_unbiased(models[model], ndraws = n, seed = reps)
                design = pdd(Y, W, D, Z, cutoff = 0.0, device = 'cuda', kernel = 'triangle')
                res_pdd = design.fit()
                
                design = rdd(Y, D, cutoff = 0.0, device = 'cuda', kernel = 'triangle')
                res_rdd = design.fit()
                
                if res_pdd.status == True and res_rdd.status == True:
                    reps += 1
                    successes += 1
                    band_pos_pdd.append(res_pdd.bandwidth['+'])
                    band_neg_pdd.append(res_pdd.bandwidth['-'])
                    band_pos_rdd.append(res_pdd.bandwidth['+'])
                    band_neg_rdd.append(res_pdd.bandwidth['-'])
                else:
                    reps += 1
                print(reps)
                print(successes)
                print('')
            band_pdd = [np.mean(band_neg_pdd), np.mean(band_pos_pdd)]
            band_rdd = [np.mean(band_neg_rdd), np.mean(band_pos_rdd)]
            covered_pdd, length_pdd, se_pdd = [], [], []
            covered_rdd, length_rdd, se_rdd = [], [], []

            while successes < band_nsims + nsims:
                Y, W, D, Z, U = sim_unbiased(models[model], ndraws = n, seed = reps)
                design = pdd(Y, W, D, Z, cutoff = 0.0, device = 'cuda', kernel = 'triangle', bandwidth = band_pdd)
                res_pdd = design.fit()
                
                design = rdd(Y, D, cutoff = 0.0, device = 'cuda', kernel = 'triangle', bandwidth = band_rdd)
                res_rdd = design.fit()
                if res_pdd.status == True and res_rdd.status == True:
                    reps += 1
                    successes += 1
                    covered_pdd.append(1 if res_pdd.est >= res_pdd.left_ci and res_pdd.est <= res_pdd.right_ci else 0)
                    covered_rdd.append(1 if res_rdd.est >= res_rdd.left_ci and res_rdd.est <= res_rdd.right_ci else 0)
                    length_pdd.append(res_pdd.right_ci - res_pdd.left_ci)
                    length_rdd.append(res_rdd.right_ci - res_rdd.left_ci)
                    se_pdd.append(res_pdd.se)
                    se_rdd.append(res_pdd.se)
                else:
                    reps += 1

            row = pd.DataFrame({'$\\tilde{\\mu}_{j}(\\cdot) = $': '\\tilde{\\mu}_{%s}(\\cdot)\\vphantom{%d}' % (model, n), 'Coverage': np.mean(covered_pdd), '$\\overline{\\text{Length}}$': np.mean(length_pdd), 
                                '$\\bar{v}_{\\text{rbc}}$': np.mean(se_pdd),  '$h_{-}$': band_pdd[0], '$h_{+}$': band_pdd[1],
                                'Coverage\\vphantom{l}': np.mean(covered_rdd), '$\\overline{\\text{Length\\vphantom{l}}}$': np.mean(length_rdd), 
                                '$\\bar{v}_{\\text{rbc}}$\\vphantom{l}': np.mean(se_rdd), '$h_{-}$\\vphantom{l}': band_pdd[0], '$h_{+}$\\vphantom{l}': band_pdd[1]}, index = [0])
            rows.append(row.set_index('$\\tilde{\\mu}_{j}(\\cdot) = $'))
        
        df_res = pd.concat(rows, axis = 0)
        df_res = df_res.apply(pd.to_numeric, errors="coerce").astype("float64")
        df_res = df_res.round(2)
        table = '\\renewcommand{\\arraystretch}{1.5}\n' + '\\begin{table}[H]\n' + \
                    '\\begin{center}\n' + '<insert caption>\n' + \
                    tabulate(df_res, headers = 'keys', tablefmt="latex_raw") + \
                    '\n\\end{center}\n' + '\\medskip\n' + '\\end{table}\n'
        table = table.replace('\\begin{tabular}{lrrrrrrrrrr}', '\\begin{tabular}{l@{\\hskip 5em}c@{\\hskip 3em}c@{\\hskip 3em}c@{\\hskip 3em}c@{\\hskip 3em}' +\
                              'c@{\\hskip 3em}c@{\\hskip 3em}c@{\\hskip 3em}c@{\\hskip 3em}c@{\\hskip 3em}c}\n' +\
                              '& \\multicolumn{5}{c}{\\hat{\\tau}_{\\text{pdd}}^{\\text{rbc}}} & ' +\
                              '\\multicolumn{5}{c}{\\hat{\\tau}_{\\text{rdd}}^{\\text{rbc}}} \\\\[-0.25em] \n' +\
                              '\\cmidrule(lr){1-5} & \\cmidrule(lr){6-11} & \\\\[-3em]')
        table = table.replace('nan', '')
        file = open(f'{outdir}/sim_no_confounding.tex', 'w')
        file.write(table)
        file.close()
                

if __name__ == '__main__':
    main()
