#include "../src/Forest.h"
#include "../src/Timer.h"

#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TSystem.h"

#include <vector>


std::vector<TString> SplitTree(const TString& in_path,
                               const TString& out_dir,
                               Int_t num_unit_entries=500,
                               const TString& tree_name="jetAnalyser") {

  TFile* in_file = TFile::Open(in_path, "READ");
  TTree* in_tree = dynamic_cast<TTree*>(in_file->Get(tree_name));
  const Int_t kInEntries = in_tree->GetEntries();

  Int_t num_fragments = std::ceil(kInEntries / float(num_unit_entries));

  TString out_name = gSystem->BaseName(in_path);
  out_name.Insert(out_name.Last('.'), "_%06d");
  TString path_fmt = gSystem->ConcatFileName(out_dir, out_name);

  std::vector<TString> out_paths;
  std::vector<TFile*> out_files;
  std::vector<TTree*> out_trees;

  for(Int_t i = 0; i < num_fragments; i++) {
    TString path = TString::Format(path_fmt, i);
    TFile* file = TFile::Open(path, "NEW");
    TTree* tree = in_tree->CloneTree(0);
    tree->SetDirectory(file);

    out_paths.push_back(path);
    out_files.push_back(file);
    out_trees.push_back(tree);
  }

  const Int_t kPrintFreq = static_cast<Int_t>( kInEntries / 10.0 );
  TString print_fmt = "[%d %] %d entry | Elapsed time: %lf sec";
  Timer timer(true);
  for(Int_t i =0; i < kInEntries; i++) {
    in_tree->GetEntry(i);

    if( i % kPrintFreq == 0 )
      std::cout << TString::Format(print_fmt, i / kPrintFreq * 10, i, timer.GetElapsedTime()) << std::endl;

    Int_t out_idx = i / num_unit_entries;
    out_trees[out_idx]->Fill();
  }

  for(TFile* each : out_files) {
    each->Write();
    each->Close();
  }

  return out_paths;
}


int main(int argc, char** argv) {
  TString in_path(argv[1]);
  const TString kTreeName = "jetAnalyser";

  TString in_dir = gSystem->DirName(in_path);
  TString out_dir = gSystem->ConcatFileName(in_dir, "Shuffled");
  if(gSystem->AccessPathName(out_dir)) {
    gSystem->mkdir(out_dir);
  }

  // Step1: Split Tree
  std::time_t time_result = std::time(nullptr);
  TString tmp_name = TString::Format("TMP_%ld", time_result);
  TString tmp_dir = gSystem->ConcatFileName(out_dir, tmp_name); 

  gSystem->mkdir(tmp_dir);

  std::vector<TString> split_result = SplitTree(in_path, tmp_dir);

  // Step3
  Forest* myforest = new Forest(split_result, "jetAnalyser"); 
  const Int_t kForestEntries = myforest->GetEntries();

  // Step4
  TString out_name = gSystem->BaseName(in_path);
  TString out_path = gSystem->ConcatFileName(out_dir, out_name);

  TFile* out_file = TFile::Open(out_path, "RECREATE");
  TTree* out_tree = myforest->CloneTree();

  myforest->CopyAddress(out_tree);

  Int_t count = 0;
  while( myforest->HasEntry() ) {
    count++;
    myforest->GetEntry();
    out_tree->Fill();
    if(count % 1000 == 0) std::cout << count << "th!" << std::endl;
  }

  out_file->Write();
  out_file->Close();

  std::cout << "Forest: " << myforest->GetEntries() << std::endl;

  return 0;
}
