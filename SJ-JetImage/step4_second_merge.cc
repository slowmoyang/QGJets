<<<<<<< HEAD
void merge(TString which, int num_files){
    TString fmt = TString::Format("./step3_shuffle/%s_%s.root", which.Data(), "%d");
    TString output_path = TString::Format("./step4_second_merge/%s.root", which.Data());
=======
#include "TString.h"
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TSystem.h"

#include<iostream>
using std::endl;
using std::cout;

void merge(TString which, int num_files){
  TString output_dir("step4_merge/");
    if(gSystem->AccessPathName(output_dir)) {
        gSystem->mkdir(output_dir);
    }

    TString fmt = TString::Format("./step3_shuffle/%s_%s.root", which.Data(), "%d");
    TString output_path = TString::Format("./step4_merge/%s.root", which.Data());
>>>>>>> upstream/master

    TChain mychain("jetAnalyser");
    for(int i=1; i <= num_files; i++){
        TString path = TString::Format(fmt, i);
        mychain.Add(path);
    }
    mychain.Merge(output_path);
}

<<<<<<< HEAD
void macro(){

    if(gSystem->AccessPathName(output_dir)){
        gSystem->mkdir(output_dir);
    }

    merge("dijet", 20);
    merge("zjet", 20);
=======
void macro();
int main()
{
  macro();
}


void macro(){
    merge("dijet", 1);
    merge("z_jet", 1);
>>>>>>> upstream/master
}
