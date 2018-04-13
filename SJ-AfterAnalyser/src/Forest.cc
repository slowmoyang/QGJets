#include "Forest.h"

// #include "BM2.h"

#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TRandom.h"

#include <vector>
// #include <cstdlib> // std::rand
#include <numeric> // std::acuumulate
#include <unordered_map>
#include <iostream>


Forest::Forest(std::vector<TString> paths, TString tree_name) {
  paths_ = paths;
  tree_name_ = tree_name;

  for(auto path : paths_) {
    TFile* root_file = TFile::Open(path, "READ");
    TTree* tree = dynamic_cast<TTree*>(root_file->Get(tree_name_));
    Int_t num_entries = tree->GetEntries();

    files_.push_back(root_file);
    trees_.push_back(tree);
    num_entries_.push_back(num_entries);
  }

  tree_candidates_.assign(trees_.begin(), trees_.end());
  num_candidates_ = tree_candidates_.size();

  entry_candidates_.resize(num_candidates_);
  std::fill(entry_candidates_.begin(), entry_candidates_.end(), 0);
}


void Forest::GetEntry() {
  Int_t tree_idx = gRandom->Uniform(num_candidates_);

  tree_candidates_[tree_idx]->GetEntry(entry_candidates_[tree_idx]); 
  entry_candidates_[tree_idx]++;

  if(entry_candidates_[tree_idx] == num_entries_[tree_idx]) {
    tree_candidates_.erase(tree_candidates_.begin() + tree_idx);
    entry_candidates_.erase(entry_candidates_.begin() + tree_idx);
    num_candidates_ = tree_candidates_.size(); 
  }
}


TTree* Forest::CloneTree() {
  return trees_[0]->CloneTree(0);
}

void Forest::CopyAddress(TTree* tree) {
  for(TTree* each : trees_) {
    tree->CopyAddresses(each);
  }
}


Bool_t Forest::HasEntry() {
    Bool_t has_entry = static_cast<Bool_t>(num_candidates_);
    return has_entry;
}


Int_t Forest::GetEntries() {
  return std::accumulate(num_entries_.begin(), num_entries_.end(), 0);
}


void Forest::Close() {
  for(auto each : files_) {
    each->Close();
  }
}
