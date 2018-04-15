#include "src/Forest.h"
#include "src/Timer.h"

#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TSystem.h"

#include <vector>
#include <iostream>


int main(int argc, char** argv) {
  const char* kEventType = argv[1];
  const TString kTreeName = "jetAnalyser";

  std::vector<TString> in_paths;

  TString path_fmt = "../Data/pt_%d_%d/3-JetImage/Shuffled/%s.root";
  for(Int_t min_pt = 100; min_pt <= 900; min_pt += 100) { 
    Int_t max_pt = min_pt + 100;
    TString path = TString::Format(path_fmt, min_pt, max_pt, kEventType);
    std::cout << path << std::endl;
    in_paths.push_back(path);
  }

  // Step3
  Forest* myforest = new Forest(in_paths, kTreeName); 
  const Int_t kForestEntries = myforest->GetEntries();

  // Step4
  TString out_dir = "../Data/pt_100_1000/3-JetImage/";
  if(gSystem->AccessPathName(out_dir)) {
    gSystem->mkdir(out_dir, true);
  }

  TString out_name = TString::Format("%s.root", kEventType);
  TString out_path = gSystem->ConcatFileName(out_dir, out_name);

  TFile* out_file = TFile::Open(out_path, "RECREATE");
  TTree* out_tree = myforest->CloneTree();

  myforest->CopyAddress(out_tree);

  TString print_fmt = "[%d/%d] %f % | Elapsed time: %.1lfsec\n";
  Timer timer(true);
  for(Int_t i = 0; i < kForestEntries; i++) {
    myforest->GetEntry();
    out_tree->Fill();
    if((i % 1000 == 0) or (i == kForestEntries)) {
        std::cout << TString::Format(print_fmt, i, kForestEntries, 100*float(i)/kForestEntries, timer.GetElapsedTime());
    }
  }

  std::cout << "Start to write output file." << std::endl;
  out_file->Write();
  std::cout << "Finish off writing output file." << std::endl;

  std::cout << "Start to close output file." << std::endl;
  out_file->Close();
  std::cout << "Finish off closing output file." << std::endl;
  return 0;
}
