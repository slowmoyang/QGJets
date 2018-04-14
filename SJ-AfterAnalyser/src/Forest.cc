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
#include <memory>
#include <cstdlib> // std::rand, std::abort


Forest::Forest(std::vector<TString> paths, TString tree_name) {
  tree_name_ = tree_name;

  Int_t idx = 0;
  num_total_entries_ = 0;
  for(auto path : paths) {

    TFile* root_file = TFile::Open(path, "READ");
    TTree* tree = dynamic_cast<TTree*>(root_file->Get(tree_name_));
    Int_t num_entries = tree->GetEntries();

    num_total_entries_ += num_entries;

    candidates_[idx] = BigTree();
    candidates_[idx].path        = path;
    candidates_[idx].file        = root_file;
    candidates_[idx].tree        = tree;
    candidates_[idx].entry       = 0;
    candidates_[idx].num_entries = num_entries;

    alives_.push_back(idx);

    idx++;
  }
}


void Forest::GetEntry() {
  // random choice
  Int_t alives_idx = std::rand() % alives_.size();
  Int_t idx = alives_[alives_idx];

  candidates_[idx].tree->GetEntry(candidates_[idx].entry);
  candidates_[idx].entry++;

  if(candidates_[idx].entry == candidates_[idx].num_entries) {
    std::cout << candidates_[idx].path << "'s tree is exhausted" << std::endl;
    alives_.erase(alives_.begin() + alives_idx);
  }
}


TTree* Forest::CloneTree() {
  return candidates_[0].tree->CloneTree(0);
}

void Forest::CopyAddress(TTree* tree) {
  for(auto each : candidates_)
    tree->CopyAddresses(each.second.tree);
}


Bool_t Forest::HasEntry() {
  Bool_t has_entry;
  if (alives_.size() > 0)       has_entry = true;
  else if (alives_.size() == 0) has_entry = false;
  else                      std::abort();
  return has_entry;
}


Int_t Forest::GetEntries() {return num_total_entries_;}


