#include <iostream>
#include <vector>
#include <algorithm>
//#include <memory>

#include "TSystem.h"
#include "TString.h"
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TDirectory.h"

#include "utils.cc"

template<typename T>
inline bool IsInfty(T i) {return std::abs(i) == std::numeric_limits<T>::infinity();}

// Step1
TString AttachLabel(const TString& input_path, const TString& output_dir){

    std::cout << "In: " << input_path << std::endl;

    // Input
    TFile* input_file = TFile::Open(input_path, "READ");
    TTree* input_tree = dynamic_cast<TTree*>( input_file->Get("jetAnalyser") );

    int kInputEntries = input_tree->GetEntries();

    TString input_name = gSystem->BaseName(input_path);

    bool is_qq = input_name.Contains("qq");
    bool is_gg = input_name.Contains("gg");
    bool is_zq = input_name.Contains("zq");

    bool is_quark_jets = is_qq or is_zq;
    bool is_dijet = is_qq or is_gg;

    TString selection;
    if( is_dijet )
        selection = "balanced";
    else
        selection = "pass_Zjets";

    // Temporary file
    /**************************************************************** 
     * Error in <TBranchElement::Fill>: Failed filling branch:deltaR, nbytes=-1
     * Error in <TTree::Fill>: Failed filling branch:jetAnalyser.deltaR, nbytes=-1, entry=10469
     *  This error is symptomatic of a Tree created as a memory-resident Tree
     *  Instead of doing:
     *      TTree *T = new TTree(...)
     *      TFile *f = new TFile(...)
     *  you should do:
     *      TFile *f = new TFile(...)
     *      TTree *T = new TTree(...)
     ***********************************************************/
    TString temp_path = input_path.Copy();
    temp_path.Insert(temp_path.Last('.'), "_TEMPORARY"); 
    TFile* temp_file = TFile::Open(temp_path, "RECREATE");
    TTree* selected_tree = input_tree->CopyTree(selection);
    selected_tree->SetDirectory(temp_file);

    const int selected_entries = selected_tree->GetEntries();

    // variables for checking infnite value.
    float ptD, axis1, axis2;
    int nmult, cmult;
    selected_tree->SetBranchAddress("ptD", &ptD);
    selected_tree->SetBranchAddress("axis1", &axis1);
    selected_tree->SetBranchAddress("axis2", &axis2);
    selected_tree->SetBranchAddress("nmult", &nmult);
    selected_tree->SetBranchAddress("cmult", &cmult);

    TString status = TString::Format(
        "NumGoodJets/NumTotalJets: %d/%d (%.2f) << std::endl",
        selected_entries,
        kInputEntries,
        float(selected_entries) / kInputEntries);
    std::cout << status << std::endl;


    // Output
    TString output_name = input_name;
    output_name.ReplaceAll("default", selection);
    TString output_path = gSystem->ConcatFileName(output_dir, output_name);

    TFile* output_file = TFile::Open(output_path, "RECREATE");
    TTree* output_tree = selected_tree->CloneTree(0);
    output_tree->SetDirectory(output_file);
    
    Int_t label = is_quark_jets ? 0 : 1;
    
    output_tree->Branch("label", &label, "label/I");
    
    for(int i=0; i < selected_entries; i++){
        selected_tree->GetEntry(i);

        if(IsInfty(cmult)) continue;
        if(IsInfty(nmult)) continue;
        if(IsInfty(ptD)) continue;
        if(IsInfty(axis1)) continue;
        if(IsInfty(axis2)) continue;

        output_tree->Fill();
    }
    
    output_file->Write();
    output_file->Close();

    temp_file->Close();    
    TString rm_temp_cmd = TString::Format("rm %s", temp_path.Data());
    gSystem->Exec(rm_temp_cmd); 

    input_file->Close();


    std::cout << "Out: " << output_path << std::endl << std::endl;
    
    return output_path;
}


TString Step1_AttachLabel(TString input_dir, TString parent_dir)
{
    TString output_dir = gSystem->ConcatFileName(parent_dir, "step1_labeling");
    if(gSystem->AccessPathName(output_dir))
        gSystem->mkdir(output_dir);

    std::vector< TString > paths = ListDir(input_dir, ".root");
    for(auto input_path : paths){
        AttachLabel(input_path, output_dir);
    }

    return output_dir;
}


// Step2
TString Step2_Merge(TString input_dir){

    TString parent_dir = gSystem->DirName(input_dir);
    TString output_dir = gSystem->ConcatFileName(parent_dir, "step2_first_merge");
    if(gSystem->AccessPathName(output_dir))
        gSystem->mkdir(output_dir);

    std::vector< TString > qq, gg, zq, zg;
    qq = ListDir(input_dir, ".root", "qq");
    gg = ListDir(input_dir, ".root", "gg");
    zq = ListDir(input_dir, ".root", "zq");
    zg = ListDir(input_dir, ".root", "zg");

    TString output_fmt = gSystem->ConcatFileName(output_dir, "%s_%d.root");

    // Dijet
    for(auto i=0; i<qq.size(); i++){
        TChain mychain("jetAnalyser");
        mychain.Add(qq[i]);
        mychain.Add(gg[i]);
        TString output_path = TString::Format(output_fmt, "dijet", i); 
        mychain.Merge(output_path);
    }

    // Z+jet
    for(auto i=0; i<zq.size(); i++){
        TChain mychain("jetAnalyser");
        mychain.Add(zq[i]);
        mychain.Add(zg[i]);
        TString output_path = TString::Format(output_fmt, "zjet", i); 
        mychain.Merge(output_path);
    }

    return output_dir;
}


/////////////////////////////////
//      STEP3
////////////////////////////////
TString ShuffleTree(const TString& input_path,
                    const TString& output_dir,
                    const TString& input_numcycle="jetAnalyser")
{

    TFile* input_file = new TFile(input_path, "READ");
    TTree* input_tree = (TTree*) input_file->Get(input_numcycle);
    const Int_t kInputEntries = input_tree->GetEntries();

    TString output_filename = gSystem->BaseName(input_path);
    TString output_path = gSystem->ConcatFileName(output_dir, output_filename);

    TFile* output_file = new TFile(output_path, "RECREATE");
    TTree* output_tree = input_tree->CloneTree(0);
    output_tree->SetDirectory(output_file);

    int order[kInputEntries];
    for(unsigned int i=0; i<kInputEntries; i++)
        order[i] = i;

    std::random_shuffle(order, order+kInputEntries );


    const int kPrintFreq = static_cast<int>( kInputEntries / 10.0 );

    for(unsigned int i=0; i < kInputEntries; i++){
        input_tree->GetEntry(order[i]);

        if((i % kPrintFreq == 0) or ( (i+1) == kInputEntries )){
            std::cout << "(" << 10 * i / kPrintFreq << "%) "
                 << i << "the entires" << std::endl;
        }

        output_tree->Fill();
    }

    output_file->Write();
    output_file->Close();

    input_file->Close();

    return output_path;
}


TString Step3_Shuffle(TString input_dir)
{

    TString parent_dir = gSystem->DirName(input_dir);
    TString output_dir = gSystem->ConcatFileName(parent_dir, "step3_shuffle");
    if(gSystem->AccessPathName(output_dir))
        gSystem->mkdir(output_dir);

    std::vector< TString > dijet_files = ListDir(input_dir, ".root", "dijet");
    std::vector< TString > zjet_files = ListDir(input_dir, ".root", "zjet");

    TString output_path;

    for(auto dj : dijet_files)
        output_path = ShuffleTree(dj, output_dir);
        std::cout << output_path << std::endl << std::endl;

    for(auto zj : zjet_files)
        output_path = ShuffleTree(zj, output_dir);
        std::cout << output_path << std::endl << std::endl;

    return output_dir;
}

//
//  STEP4 Merge
//
TString Step4_Merge(TString input_dir){

    TString parent_dir = gSystem->DirName(input_dir);
    TString output_dir = gSystem->ConcatFileName(parent_dir, "step4_second_merge");
    if(gSystem->AccessPathName(output_dir))
        gSystem->mkdir(output_dir);

    std::vector< TString > dijet_paths, zjet_paths;
    dijet_paths = ListDir(input_dir, ".root", "dijet");
    zjet_paths = ListDir(input_dir, ".root", "zjet");

    // Dijet
    TChain dijet_chain("jetAnalyser");
    TString dijet_chain_path = gSystem->ConcatFileName(output_dir, "dijet.root");
    for(auto dj : dijet_paths)
        dijet_chain.Add(dj);
    dijet_chain.Merge(dijet_chain_path);

    // Z+jet
    TChain zjet_chain("jetAnalyser");
    TString zjet_chain_path = gSystem->ConcatFileName(output_dir, "zjet.root");
    for(auto zj : zjet_paths)
        zjet_chain.Add(zj);
    zjet_chain.Merge(zjet_chain_path);

    return output_dir;
}


TString BalanceClasses(const TString& input_path, const TString& output_dir)
{
    // Input Dijet
    std::cout << "In: " << input_path << std::endl;
    TFile* input_file = TFile::Open(input_path, "READ");
    TTree* input_tree = dynamic_cast<TTree*>( input_file->Get("jetAnalyser") );
    const int kInputEntries = input_tree->GetEntries();

    Int_t label;
    input_tree->SetBranchAddress("label", &label);

    const int kNumQuark = input_tree->Draw("pt >> tmp_hist_quark", "label == 0", "goff"); 
    const int kNumGluon = input_tree->Draw("pt >> tmp_hist_gluon", "label == 1", "goff"); 

    gDirectory->Delete("tmp_hist_quark");
    gDirectory->Delete("tmp_hist_gluon");

    std::cout << "For the input," << std::endl;
    std::cout << "# of quark jets: " << kNumQuark << std::endl;
    std::cout << "# of gluon jets: " << kNumGluon << std::endl;


    // Output Dijet
    TString output_name = gSystem->BaseName(input_path);
    TString output_path = gSystem->ConcatFileName(output_dir, output_name);

    if ( kNumQuark == kNumGluon ) {
        TString mv_cmd = TString::Format("mv %s %s", input_path.Data(), output_dir.Data());
        gSystem->Exec(mv_cmd);
    }
    else {
        const int kMoreClass = kNumQuark > kNumGluon ? 0 : 1;
        const int kLessNum = kNumQuark > kNumGluon ? kNumGluon : kNumQuark;
        std::cout << "a Class with More examples: " << kMoreClass << std::endl;
        std::cout << "a Number of Less: " << kLessNum << std::endl;

        TFile* output_file = TFile::Open(output_path, "RECREATE");
        TTree* output_tree = input_tree->CloneTree(0);
        output_tree->SetDirectory(output_file);

        int count = 0;
        const int kPrintFreq = kInputEntries / 10;
        for(int i=0; i < kInputEntries; i++) {
            input_tree->GetEntry(i);

            if( i % kPrintFreq == 0 ) {
                std::cout << "(" << 10 * i / kPrintFreq << "%) "
                          << i << "th entries" << std::endl;
            }

            if( label == kMoreClass )
            {
                if ( count < kLessNum )
                {
                    output_tree->Fill();
                    count++;
                }
                else
                    continue;
            }
            else
            {
                output_tree->Fill();
            }
        }

        const int kNumOutQuark = output_tree->Draw("pt >> tmp_hist_quark", "label == 0", "goff"); 
        const int kNumOutGluon = output_tree->Draw("pt >> tmp_hist_gluon", "label == 1", "goff"); 

        gDirectory->Delete("tmp_hist_quark");
        gDirectory->Delete("tmp_hist_gluon");

        std::cout << std::endl;
        std::cout << "For the output" << std::endl;
        std::cout << "# of quark jets: " << kNumOutQuark << std::endl;
        std::cout << "# of gluon jets: " << kNumOutGluon << std::endl;


        output_file->Write();
        output_file->Close();
    }

    std::cout << "Out: " << output_path << std::endl;
    return output_path;
}



TString Step5_BalanceClasses(const TString& input_dir)
{
    TString output_dir = gSystem->DirName(input_dir);

    TString dijet_path = gSystem->ConcatFileName(input_dir, "dijet.root");
    TString zjet_path = gSystem->ConcatFileName(input_dir, "zjet.root");

    BalanceClasses(dijet_path, output_dir);
    BalanceClasses(zjet_path, output_dir);
    return output_dir; 

}


void macro(const TString& input_dir){
 
    TString parent_dir = gSystem->DirName(input_dir);
    TString merged_dir = gSystem->ConcatFileName(parent_dir, "2-Merged");
    if(gSystem->AccessPathName(merged_dir))
        gSystem->mkdir(merged_dir);
   
    std::cout << "Step 1: Select good jets and attach label" << std::endl << std::endl;
    TString step1_path = Step1_AttachLabel(input_dir, merged_dir);

    std::cout << "Step 1: Merge one quark jet and gluon jet file." << std::endl << std::endl;
    TString step2_path = Step2_Merge(step1_path);

    std::cout << "Step 3: Shuffle small dataset." << std::endl << std::endl;
    TString step3_path = Step3_Shuffle(step2_path);

    std::cout << "Step 4: Merge the entire dataset." << std::endl << std::endl;
    TString step4_path = Step4_Merge(step3_path);

    std::cout << "Step 4: Balance classes in a dataset" << std::endl << std::endl;
    TString step5_path = Step5_BalanceClasses(step4_path);
}


int main(int argc, char *argv[])
{
    TString input_dir = argv[1];
    macro(input_dir);
    return 0;
}