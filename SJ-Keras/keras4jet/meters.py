import os
import ROOT
import numpy as np
from sklearn.metrics import roc_curve, auc
from statsmodels.nonparametric.smoothers_lowess import lowess

import matplotlib.pyplot as plt


ROOT.gROOT.SetBatch(True)


class Meter(object):
    def __init__(self, name_list, dpath):
        self.data = {name: np.empty(shape=(0,), dtype=np.float32) for name in name_list}
        self.dpath = dpath
        self.plot_list = []

    def append(self, data_dict):
        for k in data_dict.keys():
            self.data[k] = np.append(arr=self.data[k], values=data_dict[k])
        
    def add_plot(self, x, ys, title, xlabel, ylabel, use_lowess=True):
        """
        ys: [(name, label, color) or (name, label, color)]
        """
        ys = map(self.normalize_ys, ys)
        self.plot_list.append({
            "x": x,
            "ys": ys,
            "title": title,
            "xlabel": xlabel,
            "ylabel": ylabel,
            "use_lowess": use_lowess})

        
    def plot(self, plot_info):
        plt.figure(figsize=(8, 6))
        plt.rc("font", size=12)

        x = self.data[plot_info["x"]]

        for yname, label, color in plot_info["ys"]:
            y = self.data[yname]
            plt.plot(x, y, label=label, color=color, lw=2, alpha=0.2)

            if plot_info["use_lowess"]:
                # smooth 
                filtered = lowess(y, x, is_sorted=True, frac=0.075, it=0)
            
                plt.plot(filtered[:,0], filtered[:,1],
                         label=label+"(LOWESS)", color=color, lw=2, alpha=1)

        plt.xlabel(plot_info["xlabel"])
        plt.ylabel(plot_info["ylabel"])

        plt.title(plot_info["title"])
        plt.legend(loc='best')
        plt.grid()

        filename = plot_info["title"].replace(" ", "_") + ".png"
        path = os.path.join(self.dpath, filename)
        plt.savefig(path)
        plt.close()
        

    def save(self):
        path = os.path.join(self.dpath, "validation.npz")
        np.savez(path, **self.data)
        
    def finish(self):
        for plot_info in self.plot_list:
            self.plot(plot_info)
        self.save() 


    def normalize_ys(self, y_info):
        length = len(y_info)
        if length == 2:
            color = self._color(y_info[0])
            y_info += (color,)
        elif length > 3 or length <= 1:
            raise ValueError
        return y_info
        
    def _color(self, y):
        if "train" in y:
            color = "navy"
        elif "dijet" in y:
            color = 'orange'
        elif "zjet" in y:
            color = "indianred"
        else:
            color = np.random.rand(3,1)
        return color


class ROCMeter(object):
    def __init__(self, dpath, step, title, prefix=""):
        self.dpath = dpath
        self.step = step
        self.title = title
        self._prefix = prefix

        self.y_true = np.array([])
        self.y_pred = np.array([])  # predictions

        # uninitialized attributes
        self.fpr = None
        self.tpr = None
        self.fnr = None
        self.auc = None

    def append(self, y_true, y_pred):
        # self.y_true = np.r_[self.y_true, y_true]
        # self.y_pred = np.r_[self.y_pred, y_pred]
        self.y_true = np.append(self.y_true, y_true[:,1])
        self.y_pred = np.append(self.y_pred, y_pred[:, 1])


    def compute_roc(self):
        self.fpr, self.tpr, _ = roc_curve(self.y_true, self.y_pred)
        self.fnr = 1 - self.fpr
        self.auc = auc(self.fpr, self.tpr)

    def save_roc(self, path):
        logs = np.vstack([self.tpr, self.fnr, self.fpr]).T
        np.savetxt(path, logs, delimiter=',', header='tpr, fnr, fpr')

    def plot_roc_curve(self, path):
        # fig = plt.figure()
        plt.plot(self.tpr, self.fnr, color='darkorange',
                 lw=2, label='ROC curve (area = {:0.3f})'.format(self.auc))
        plt.plot([0, 1], [1, 1], color='navy', lw=2, linestyle='--')
        plt.plot([1, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.1])
        plt.ylim([0.0, 1.1])
        plt.xlabel('Quark Jet Efficiency (TPR)')
        plt.ylabel('Gluon Jet Rejection (FNR)')
        plt.title('{}-{} / ROC curve'.format(self.title, self.step))
        plt.legend(loc='lower left')
        plt.grid()
        plt.savefig(path)
        plt.close()

    def finish(self):
        self.compute_roc()

        filename_format = '{prefix}roc_step-{step}_auc-{auc:.3f}.{ext}'.format(
            prefix=self._prefix,
            step=str(self.step).zfill(6),
            auc=self.auc,
            ext='%s'
        )

        csv_path = os.path.join(self.dpath, filename_format % 'csv')
        plot_path = os.path.join(self.dpath, filename_format % 'png')

        self.save_roc(csv_path)
        self.plot_roc_curve(plot_path)






class OutHist(object):
    def __init__(self, dpath, step, dname_list):
        filename = "outhist-step_{step}.root".format(step=step)
        path = os.path.join(dpath, filename)

        self._dname_list = dname_list

        self.root_file = ROOT.TFile(path, "RECREATE")
        self.hists = {}
        for dname in dname_list:
            self.make_dir_and_qg_hists(dname)

    def make_dir_and_qg_hists(self, dname):
        # root file
        self.root_file.mkdir(dname)
        self.root_file.cd(dname)
        # just attributes
        self.hists[dname] = {}
        # histograms
        self.hists[dname]["quark"] = self._make_hist("quark")
        self.hists[dname]["quark"].SetDirectory(self.root_file.Get(dname))
        self.hists[dname]["gluon"] = self._make_hist("gluon")
        self.hists[dname]["gluon"].SetDirectory(self.root_file.Get(dname))


    def _make_hist(self, name):
        return ROOT.TH1F(name, name, 100, 0, 1)

    def cd(self, dname):
        self.root_file.cd(dname)

    def fill(self, dname, y_true, y_pred):
        for is_gluon, gluon_likeness in zip(y_true[:, 1], y_pred[:, 1]):
            if is_gluon:
                self.hists[dname]["gluon"].Fill(gluon_likeness)
            else:
                self.hists[dname]["quark"].Fill(gluon_likeness)

    def save(self):
        self.root_file.Write()
        self.root_file.Close()

    def finish(self):
        self.save()
