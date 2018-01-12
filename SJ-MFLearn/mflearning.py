from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os 
import time
from datetime import datetime
import argparse

import ROOT

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import offsetbox
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from sklearn import manifold
from sklearn import decomposition
from sklearn import ensemble
from sklearn import discriminant_analysis
from sklearn import random_projection


def timeit(method):
    def timed(*args, **kw):
        start = time.time()
        result = method(*args, **kw)
        end = time.time()
        exec_time = end - start
        print("{exec_time:.2f} sec".format(exec_time=exec_time))
        return result
    return timed


class ManifoldLearner(object):
    """
    ref. http://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html#sphx-glr-auto-examples-manifold-plot-lle-digits-py
    """
    def __init__(self, input_path, output_path, switches,
                 n_components=2, n_neighbors=4,
                 random_state=0):
        self._input_path = input_path
        self._output_path = output_path
        self._n_components = n_components

        n_neighbors_min = np.ceil(n_components * (n_components + 3) / 2).astype(int) + 1
        if n_neighbors >= n_neighbors_min:
            self._n_neighbors = n_neighbors
        else:
            self._n_neighbors = n_neighbors_min



        self._switches = switches
        self._random_state = random_state

        self._X_image, self._X_vars, self._y, self._info = self.load_data(self._input_path)

        self._X = self._X_vars
        self._data = "jet discriminating variables"

        self._collection = {}
        self._collection["X"] = self._X
        self._collection["y"] = self._y

        self.run()

    def run(self):
        for name, v in self._switches.iteritems():
            if not v:
                continue

            X_emb, title = getattr(self, name)()
            print("X_emb shape: {shape}".format(
                shape=X_emb.shape))
            self.plot_embedding(
                X_emb=X_emb, info=self._info, title=title)
            self._collection[name] = X_emb
            print("\n{line}\n".format(line="#"*20))


        # save data
        npz_path = os.path.join(self._output_path, "collection.npz")
        np.savez(npz_path, **self._collection)

 
    def load_data(self, input_path):
    	def _color(partonId, nMatchedJets):
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

    	def _symbol(partonId):
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


        f = ROOT.TFile(input_path, "READ")
        key = f.GetListOfKeys().At(0).GetName()
        tree = f.Get(key)
 
        num_examples = tree.GetEntries()
        num_features = len(tree.image)

        X_image = []
        X_vars = []
        y = []
        info = []
        for i in xrange(num_examples):
            tree.GetEntry(i)

            nMatchedJets = tree.nMatchedJets
            if not (nMatchedJets in [2, 3]):
                continue

            partonId = tree.partonId

            X_image.append(np.array(tree.image))
            X_vars.append(np.array(tree.variables))

            y.append(tree.label[1])

            color = _color(partonId, nMatchedJets)
            symbol = _symbol(partonId)

            info.append(
                {"partonId": partonId,
                 "nMatchedJets": nMatchedJets,
                 "color": color,
                 "symbol": symbol}
            )

        X_image = np.array(X_image)
        X_vars = np.array(X_vars)
        y = np.array(y)
        return X_image, X_vars, y, info


    def plot_embedding(self, X_emb, info, title):
    	def _get_artist(marker_str, color):
	        marker = "${marker_str}$".format(marker_str=marker_str)
    	    artist = Line2D(
	        	range(1), range(1),
                color="white",
        		marker=marker,
        		markeredgecolor=color,
                markerfacecolor=color,
        		markeredgewidth=1,
                markersize=15)
    	    return artist
	
    	fig = plt.figure(figsize=(10, 10))
    	ax = fig.add_subplot(111)

    	# Scale the embedding vectors
    	X_emb[:, 0] /= np.abs(X_emb[:, 0]).max()
    	X_emb[:, 1] /= np.abs(X_emb[:, 1]).max()

    	for i in range(X_emb.shape[0]):
	        plt.text(
        		x=X_emb[i][0],
                y=X_emb[i][1],
    	    	s=info[i]['symbol'],
    	    	fontdict={'weight': "bold", "size": 9},
                color=info[i]['color'], 
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

        filename = "{title}.{ext}".format(
                title=title.replace(" ", "_"),
                ext="png")
        path = os.path.join(self._output_path, filename)
    	plt.savefig(path)


    @timeit
    def random_projection(self):
        """
        Random 2D projection using a random unitary matrix
        """
        print("Computing random projection")
        rp = random_projection.SparseRandomProjection(
            n_components=self._n_components,
            random_state=self._random_state)
        X_projected = rp.fit_transform(X=self._X)
        return X_projected, "Random Projection"

    @timeit
    def PCA(self):
        """
        Projection on to the first 2 principal components
        """
        print("Computing PCA projection")
        pca = decomposition.TruncatedSVD(
            n_components=self._n_components)
        X_pca = pca.fit_transform(X=self._X)
        return X_pca, "PCA projection"

    @timeit
    def LDA(self):
        """Projection on to the first n linear discriminant components"""
        print("Computing Linear Discriminant Analysis projection")
        X2 = self._X.copy()
        X2.flat[::self._X.shape[1] + 1] += 0.01  # Make X invertible
        lda = discriminant_analysis.LinearDiscriminantAnalysis(
            n_components=self._n_components)
        X_lda= lda.fit_transform(X=X2, y=self._y)
        return X_lda, "Lienar discriminant analysis projection"

    @timeit
    def isomap(self):
        """
        Isomap projection
          - Non-linear dimensionality reduction through Isometric Mapping
        """
        print("Computing Isomap embedding")
        iso = manifold.Isomap(
            n_neighbors=self._n_neighbors,
            n_components=self._n_components)
        X_iso = iso.fit_transform(X=self._X)
        return X_iso, "Isomap projection"

    def _LLE(self, method):
        clf = manifold.LocallyLinearEmbedding(
            n_neighbors=self._n_neighbors,
            n_components=self._n_components,
            method=method
        )
        return clf
 
    @timeit
    def LLE(self):
        """
        Locally linear embedding
        """
        clf = self._LLE("standard")
        X_lle = clf.fit_transform(X=self._X)
        return X_lle, "LLE embedding"

    @timeit
    def MLLE(self):
        """
        MLLE
          - Modified locally linear embedding
          - LLE with the modified locally linear embedding algorithm
        """
        clf = self._LLE("modified")
        X_mlle = clf.fit_transform(X=self._X)
        return X_mile, "Modified LLE embedding"

    @timeit
    def HLLE(self):
        """
        HLLE embedding
          _ HLLE: Hessian Locally Linear Embedding
        """
        print("Computing Hessian LLE embedding")
        clf = self._LLE("hessian")
        X_hlle = clf.fit_transform(X=self._X)
        return X_hlle, "Hessian LLE embedding"

    @timeit
    def LTSA(self):
        """
        LTSA embedding
          - LTSA: local tangent space alignment algorithm
        """
        print("Computing LTSA embedding")
        clf = self._LLE("ltsa")
        X_ltsa = clf.fit_transform(X=self._X)
        return X_ltsa, "LTSA embedding"

    @timeit
    def MDS(self):
        """
        MDS  embedding
          - Multidimensional scaling ebedding
        """
        print("Computing MDS embedding")
        clf = manifold.MDS(
            n_components=self._n_components,
            n_init=1,
            max_iter=100
        )
        X_mds = clf.fit_transform(X=self._X)
        return X_mds, "MDS ebedding"


    @timeit
    def random_trees(self):
        """
        Random Trees embedding
          - The RandomTreesEmbedding, from the sklearn.ensemble module, is not
            technically a manifold embedding method, as it learn a high-
            dimensional representation on which we apply a dimensionality
            reduction method. However, it is often useful to cast a dataset
            into a representation in which the classes are linearly-separable.
        """
        print("Computing Totally Random Trees embedding")
        hasher = ensemble.RandomTreesEmbedding(
            n_estimators=200,
            random_state=self._random_state,
            max_depth=5
        )
        X_transformed = hasher.fit_transform(X=self._X)
        pca = decomposition.TruncatedSVD(
            n_components=self._n_components
        )
        X_reduced = pca.fit_transform(X_transformed)
        return X_reduced, "Totally random trees embedding"


    @timeit
    def spectral(self):
        """Spectral embedding"""
        print("Computing Spectral embedding")
        embedder = manifold.SpectralEmbedding(
            n_components=self._n_components,
            random_state=self._random_state,
            eigen_solver="arpack"
        )
	X_se = embedder.fit_transform(X=self._X)
        return X_se, "Spectral embedding"

    @timeit
    def t_SNE(self):
        """
        t-SNE embedding
          - t-distributed stochastic neighbor embedding
          - https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding
          - t-SNE will be initialized with the embedding that is generated by
            PCA in this example, which is not the default setting. It ensures
            global stability of the embedding, i.e., the embedding does not
            depend on random initialization.
        """
        print("Computing t-SNE embedding")
        tsne = manifold.TSNE(
            n_components=self._n_components,
            init='pca',
            random_state=self._random_state
        )
        X_tsne = tsne.fit_transform(X=self._X)
        return X_tsne, "t-SNE embedding"

    def set_X(self, X):
        if isinstance(X, np.ndarray):
            self._X = X
        elif isinstance(X, str):
            if X.lower() == "image":
                self._X = self._X_image
            elif X.lower() == "variables":
                self._X = self._X_vars
        else:
            raise NotImplementedError("")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str,
                        default="../data/root_format/dataset_13310/training_weak_7986.root",
                        help="the input path")

    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%s")
    output_path_default = os.path.join(
        "./logs", "log-{now}".format(now=now))
    parser.add_argument("--output_path", type=str,
                        default=output_path_default,
                        help="the output path")

    parser.add_argument("--n_components", type=int,
                        default=2,
                        help="the number of components")
    parser.add_argument("--n_neighbors", type=int,
                        default=4,
                        help="the number of neighbors")

    # switch
    parser.add_argument(
        "--random_projection", type=bool, default=True,
        help="random projection") # help
    parser.add_argument(
        "--PCA", type=bool, default=True,
        help="Principal component analysis")
    parser.add_argument(
        "--LDA", type=bool, default=True,
        help="Linear Discriminant Analysis")
    parser.add_argument(
        "--isomap", type=bool, default=True,
        help="Non-linear dimensionality reduction through Isometric Mapping")
    parser.add_argument(
        "--LLE", type=bool, default=True,
        help="Locally linear embedding")
    parser.add_argument(
        "--MLLE", type=bool, default=False,
        help="LLE with the modified LLE algorithm")
    parser.add_argument(
        "--HLLE", type=bool, default=False,
        help="LLE with the Hessian eigenmap method")
    parser.add_argument(
        "--LTSA", type=bool, default=False,
        help="LLE with local tangent space alignment algorithm")
    parser.add_argument(
        "--MDS", type=bool, default=True,
        help="Multidimensional scaling")
    parser.add_argument(
        "--random_trees", type=bool, default=True,
        help="totally random trees embedding")
    parser.add_argument(
        "--spectral", type=bool, default=True,
        help="Spectral embedding for non-linear dimensionality reduction")
    parser.add_argument(
        "--t_SNE", type=bool, default=True,
        help="t-distributed stochastic neighbor embedding")
    args = parser.parse_args()

    args_dict = vars(args)

    switches = dict(filter(lambda (k,v): isinstance(v, bool), args_dict.iteritems()))

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    ManifoldLearner(
        input_path=args.input_path,
        output_path=args.output_path,
        n_components=args.n_components,
        n_neighbors=args.n_neighbors,
        switches=switches
    )


if __name__ == "__main__":
    main()
