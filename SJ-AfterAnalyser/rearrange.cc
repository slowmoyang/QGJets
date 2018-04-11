#include "src/Timer.h"

#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TSystem.h"
#include "TChain.h"

#include <iostream>
#include <vector>

void MergeTrees(std::vector<TString> paths,
                TString out_path,
                TString tree_name="jetAnalyser") {
  std::cout << "In: " << std::endl;
  for(auto each : paths) {
    std::cout << "\t" << each << std::endl;
  }

  TChain mychain(tree_name);
  for(auto each : paths) {
    mychain.Add(each);
  }

  mychain.Merge(out_path);

  std::cout << "Out: " << out_path << std::endl;
}


inline Int_t ConvertPt2Idx(Float_t pt) { return  static_cast<Int_t>(pt / 100.0) - 1;}


int main(int argc, char** argv) {
    const TString kTreeName = "jetAnalyser";
    TString event_type(argv[1]);

    TString in_fmt = "../Data/root_%d_%d/3-JetImage/%s.root";
    TString out_fmt = "../Data/pt_%d_%d/3-JetImage/%s.root";

    std::vector<TString> in_paths, out_paths;
    for(Int_t min_pt=100; min_pt <= 900; min_pt += 100) {
        Int_t max_pt = min_pt + 100;
        TString in_path = TString::Format(in_fmt, min_pt, max_pt, event_type.Data());
        in_paths.push_back(in_path);

        TString out_path = TString::Format(out_fmt, min_pt, max_pt, event_type.Data());
        out_paths.push_back(out_path);

        TString out_dir = gSystem->DirName(out_path);
        gSystem->mkdir(out_dir, true);
    }

    TString merged_path = TString::Format("../Data/Deprecated/merged_%s.root", event_type.Data());
    MergeTrees(in_paths, merged_path); 

    std::cout << "Start to read merged file" << std::endl;
    TFile* merged_file = TFile::Open(merged_path, "READ");
    TTree* merged_tree = dynamic_cast<TTree*>(merged_file->Get(kTreeName));
    Float_t pt = 0.0;
    merged_tree->SetBranchAddress("pt", &pt);
    std::cout << "Finish off reading merged file" << std::endl;


    std::cout << "Start to create output files and trees." << std::endl;
    std::vector<TFile*> out_files;
    std::vector<TTree*> out_trees;
    for(auto out_p : out_paths) {
        TFile* out_f = TFile::Open(out_p, "RECREATE");
        TTree* out_t = merged_tree->CloneTree(0);
        out_files.push_back(out_f);
        out_trees.push_back(out_t);
    }
    std::cout << "Finish off creating output files and trees." << std::endl;

    const int kMergerEntries = merged_tree->GetEntries();
    const Int_t kPrintFreq = static_cast<Int_t>( kMergerEntries / 10 );
    TString print_fmt = "[%d %] %d entry | Elapsed time: %lf sec";

    Timer timer(true);
    for(Int_t i = 0; i< kMergerEntries; i++) {
        if( i % kPrintFreq == 0 ) {
            std::cout << TString::Format(print_fmt, i / kPrintFreq * 10, i, timer.GetElapsedTime()) << std::endl;
        }
        merged_tree->GetEntry(i);
        if((pt < 100) or (pt > 1000)) continue;


        Int_t idx = ConvertPt2Idx(pt);
        out_trees[idx]->Fill();
    }

    for(auto out_f : out_files) {
        out_f->Write();
        out_f->Close();
    }

    merged_file->Close();
    TString rm_merged_cmd = TString::Format("rm -f %s", merged_path.Data());
    gSystem->Exec(rm_merged_cmd);
    
    return 0;
}
