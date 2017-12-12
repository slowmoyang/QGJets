#include "TString.h"
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TSystem.h"
#include "TRandom.h"

#include<iostream>
using std::endl;
using std::cout;

void split(TString which, float train_split=0.8){
    TString output_dir("step5/");
    if(gSystem->AccessPathName(output_dir)) {
        gSystem->mkdir(output_dir);
    }

    TString in_path = TString::Format("./step4_merge/%s.root", which.Data());
    TString output_path_train = TString::Format("./step5/%s_train.root", which.Data());
    TString output_path_test = TString::Format("./step5/%s_test.root", which.Data());

    TFile in_f(in_path.Data(), "read");
    auto input_tree = (TTree*) in_f.Get("jetAnalyser");

    TFile output_file_train(output_path_train, "recreate");
    TTree* output_tree_train = input_tree->CloneTree(0);
    output_tree_train->SetDirectory(&output_file_train);

    TFile output_file_test(output_path_test, "recreate");
    TTree* output_tree_test = input_tree->CloneTree(0);
    output_tree_test->SetDirectory(&output_file_test);

    for (int i = 0; i < input_tree->GetEntries(); ++i) {
      input_tree->GetEntry(i);
      if (gRandom->Uniform(0, 1) < train_split)
	output_tree_train->Fill();
      else
	output_tree_test->Fill();
    }

    output_tree_train->Write();
    output_tree_test->Write();

    output_file_train.Write();
    output_file_train.Close();
    output_file_test.Write();
    output_file_test.Close();
}

void macro();
int main()
{
  macro();
}


void macro(){
    split("dijet", 0.8);
    split("z_jet", 0.8);
}
