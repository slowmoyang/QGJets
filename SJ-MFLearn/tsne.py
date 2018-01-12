import ROOT
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn import manifold
from time import time


def get_color(partonId, nMatchedJets):
    if partonId == 21:
        if nMatchedJets == 2:
            return "lightcoral"
        elif nMatchedJets == 3:
            return "red"
        else:
            raise NotImplementedError("")
    else:
        if nMatchedJets == 2:
            return "skyblue"
        elif nMatchedJets == 3:
            return "blue"
        else:
            raise NotImplementedError("")

def get_symbol(partonId):
    if abs(partonId) == 1:
        symbol = 'u'
    elif abs(partonId) == 2:
        symbol = "d"
    elif abs(partonId) == 3:
        symbol = "s"
    elif partonId == 21:
        symbol = "g"
    else:
        raise NotImplementedError("")

    if partonId < 0:
        symbol += "-"

    return symbol

def convert_tree_to_np(tree):
    num_examples = tree.GetEntries()
    num_features = len(tree.image)

    X = []
    y = []
    for i in xrange(num_examples):
        tree.GetEntry(i)

        nMatchedJets = tree.nMatchedJets
        if not (nMatchedJets in [2, 3]):
            continue

        partonId = tree.partonId

        X.append(np.array(tree.image))

        y.append({"partonId": partonId,
                  "nMatchedJets": nMatchedJets,
                  "color": get_color(partonId, nMatchedJets),
                  "symbol": get_symbol(partonId)})

    return X, y


def plot_embedding(X_emb, y, title):

    def _get_artist(marker_str, color):
        marker = "${marker_str}$".format(marker_str=marker_str)

        artist = Line2D(
            range(1), range(1), color="white",
            marker=marker,
            markeredgecolor=color, markerfacecolor=color,
            markeredgewidth=1, markersize=15
        )
        return artist
    


    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    # Scale the embedding vectors
    X_emb[:, 0] /= np.abs(X_emb[:, 0]).max()
    X_emb[:, 1] /= np.abs(X_emb[:, 1]).max()

    for i in range(X_emb.shape[0]):
        plt.text(
            x=X_emb[i][0], y=X_emb[i][1],
            s=y[i]['symbol'], color=y[i]['color'],
            fontdict={'weight': "bold", "size": 9},
            alpha=0.75
        )

    artist1 = _get_artist(marker_str="g", color="lightcoral")
    label1 = "gluon with nMatchedJets=2"

    artist2 = _get_artist(marker_str="g", color="red")
    label2 = "gluon with nMatchedJets=3"

    artist3 = _get_artist(marker_str="q", color="skyblue")
    label3 = "quark with nMatchedJets=2"

    artist4 = _get_artist(marker_str="q", color="blue")
    label4 = "quark with nMatchedJets=3"

    artist_list = [artist1, artist2, artist3, artist4]
    labels_list = [label1, label2, label3, label4]

    plt.legend(artist_list, labels_list, numpoints=1, loc=1)

    ax.set_title(label=title, fontdict={'fontsize': 20})
    ax.set_xlim(-1.0 , 1.0)
    ax.set_ylim(-1.0 , 1.0)

    plt.savefig(title+".png")






def main():
    f = ROOT.TFile("../data/root_format/dataset_13310/test_2662.root", "READ")
    tree = f.Get("jet")
    X, y = convert_tree_to_np(tree)

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X)

    plot_embedding(X_emb=X_tsne, y=y, title="t-SNE")


if __name__ == "__main__":
    main()
