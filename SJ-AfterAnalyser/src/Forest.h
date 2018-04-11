#ifndef FOREST_H
#define FOREST_H

#include "TFile.h"
#include "TTree.h"
#include "TString.h"

#include <vector>
// #include <cstdlib>
#include <numeric>
#include <unordered_map>

class Forest {
public:
  Forest(std::vector<TString> paths, TString tree_name);
  ~Forest();

  TTree* CloneTree();
  void CopyAddress(TTree* tree);
  Int_t GetEntry();
  Int_t GetEntries();
  Bool_t HasEntry();
  void Close();

private:
  std::vector<TString> paths_;
  TString tree_name_;
  std::vector<TFile*>  files_;
  std::vector<TTree*>  trees_;
  std::vector<Int_t>   num_entries_;

  std::vector<TTree*> tree_candidates_;
  std::vector<Int_t> entry_candidates_;
  Int_t num_candidates_;
};

#endif
