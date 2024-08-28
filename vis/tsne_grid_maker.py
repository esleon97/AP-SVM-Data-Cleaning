import time, pickle, argparse, tqdm
import lgdo
from sklearn.manifold import TSNE

'''
Make a grid of different TSNE hyperparameter combinations
'''

argparser = argparse.ArgumentParser()
argparser.add_argument("--file", help="file containing training data", type=str, required=True)
args = argparser.parse_args()

#Load data
sto = lgdo.lh5.LH5Store()
tb_dsp, _ = sto.read('detector/dsp', args.file)

dwts_norm = tb_dsp['dwt_norm'].nda
labels = tb_dsp['dc_label'].nda


# Make TSNE grid 

perplex_range            = [30,90,150,210]
learning_rate_range      = [200,400,600,800]
tsne_plotter             = []

print("Staring TSNE grid data production...")
start_time = time.time()
for p in tqdm.tqdm(perplex_range):
    for lr in learning_rate_range:
        tsne = TSNE(n_components=3, 
                    perplexity = p,
                    early_exaggeration= 18.0,
                    random_state=0,
                    learning_rate= lr, 
                    metric = 'chebyshev',
                    init = "pca", 
                    n_jobs = -1)
        dwts_norm_3d = tsne.fit_transform(dwts_norm)
        tsne_plotter.append((p, lr, dwts_norm_3d))
print("--- %s minutes elapsed ---" % ((time.time() - start_time)/60))

# Save TSNE grid data

with open("../data/tsne_grid_data.sav", "wb") as output:
    pickle.dump(tsne_plotter, output)

