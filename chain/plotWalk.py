import pickle
import matplotlib.pyplot as plt
import pathlib

plt.rcParams['image.cmap'] = 'Blues'
plt.rcParams['font.family'] = 'monospace'

def PlotAllWalkers(sample_chain, yAxisLabel, figtitle):
    n = len(yAxisLabel)
    fig,ax = plt.subplots(n,1,figsize=(8, n*3))
    nWalkers = sample_chain.shape[0]
    for j in range(len(yAxisLabel)):
        for i in range(nWalkers):
            ax[j].plot(sample_chain[i,:,j], color='0.5')

    [ax[i].set_xlabel('Step number') for i in range(len(yAxisLabel))]
    [ax[i].set_ylabel(yAxisLabel[i]) for i in range(len(yAxisLabel))]
    ax[0].set_title(figtitle)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    parser = ArgumentParser(description=(
        """
        script to construct ePSF from an exposure

        """), formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument("--chainfile", type=str, help="name of chain pickle file")
    parser.add_argument("--walkfile", type=str, help="plot of walks file name")
    parser.add_argument("--chaindir", type=str, default="chainResults")
    parser.add_argument("--modelNum", type=int)
    args = parser.parse_args()
    filepath = pathlib.Path.joinpath(pathlib.Path(args.chaindir), args.chainfile)
    with open(filepath, "rb") as file:
        c = pickle.load(file)
    labels = {}
    labels[0] = ['X0_1','Y0_1','I_tot_1', 'PA_2','ell_bulge_2','I_e_2', 'r_e_2']
    labels[1] = ['X0_1', 'Y0_1', 'I_tot_1', 'PA_2', 'ell_bulge_2', 'I_e_2','r_e_2', 'X0_2', 'Y0_2', 'I_tot_3', 'PA_4', 'ell_bulge_4', 'I_e_4', 'r_e_4']
    fig = PlotAllWalkers(c['chain'], labels[args.modelNum], "Walkers n1, model "+str(args.modelNum))
    savepath = pathlib.Path.joinpath(pathlib.Path(args.chaindir), args.walkfile)
    fig.savefig(savepath)
