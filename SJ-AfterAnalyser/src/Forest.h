#ifndef FOREST_H
#define FOREST_H

#include "TFile.h"
#include "TTree.h"
#include "TString.h"

#include <vector>
#include <map>
// #include <cstdlib>
#include <numeric>
#include <unordered_map>

struct BigTree {
  TString path;
  TFile*   file;
  TTree*   tree;
  Int_t    entry;
  Int_t    num_entries;
};


class Forest {
public:
  Forest(std::vector<TString> paths, TString tree_name);

  TTree* CloneTree();
  void CopyAddress(TTree* tree);
  void GetEntry();
  Int_t GetEntries();
  Bool_t HasEntry();

private:
  TString tree_name_;
  std::map<Int_t, BigTree> candidates_;
  std::vector<Int_t> alives_;
  Int_t num_alive_;

  Int_t num_total_entries_;
};

#endif
