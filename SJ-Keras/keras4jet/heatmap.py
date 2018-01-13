import os
import ROOT
import numpy as np 

class Heatmap(object):

    def __init__(self, data_set, out_dir, nbins=25, cmap="COLZTEXT"):
        self.names = ["total", "correct", "incorrect", "scaled_correct", "scaled_incorrect"]

        self.get_pt_eta_range(data_set)


        self.out_path_fmt = os.path.join(out_dir, "{name}.{ext}")
        self.out_file = ROOT.TFile.Open(
            self.out_path_fmt.format(name="heatmap", ext="root"), "RECREATE")

        self.heatmap = {}
        for name in self.names:
            self.heatmap[name] = ROOT.TH2F(
                name, name,
                nbins, self.pt_min, self.pt_max,
                nbins, self.eta_min, self.eta_max)
            self.heatmap[name].SetDirectory(self.out_file)


        self.data_set = data_set
        self.nbins = nbins
        self.cmap = cmap

    def get_pt_eta_range(self, path, treename="jetAnalyser"):
        f = ROOT.TFile.Open(path, "READ")
        tree = f.Get(treename)
        self.pt_min = tree.GetMinimum("pt")
        self.pt_max = tree.GetMaximum("pt")
        self.eta_min = -2.4
        self.eta_max = 2.4
        del tree
        f.Close()
        del f
        return None


    def fill(self, y_true, y_pred, pt, eta):
        y_true = y_true[:, 1]
        y_pred = np.where(y_pred[:, 1] > 0.5, 1, 0)
        correct = np.equal(y_true, y_pred) 
        for c, p, e in zip(correct, pt, eta):
            self.heatmap["total"].Fill(p, e)
            if c:
                self.heatmap["correct"].Fill(p, e)
            else:
                self.heatmap["incorrect"].Fill(p, e)

    def plot(self, name):
        can = ROOT.TCanvas(name, name, 1200, 600)
        if "scaled" in name:
            ROOT.gStyle.SetPaintTextFormat(".2f")
        ROOT.gStyle.SetOptStat(0)
        self.heatmap[name].Draw(self.cmap)
        self.heatmap[name].GetXaxis().SetTitle("pT (GeV)")
        self.heatmap[name].GetYaxis().SetTitle("\eta")
        out_path = self.out_path_fmt.format(name=name, ext="png")
        can.SaveAs(out_path)

    def finish(self):
        self.heatmap["scaled_correct"].Divide(
            self.heatmap["correct"], self.heatmap["total"])

        self.heatmap["scaled_incorrect"].Divide(
            self.heatmap["incorrect"], self.heatmap["total"])

        for name in self.names:
            self.plot(name)

        self.out_file.Write()
        self.out_file.Close()
