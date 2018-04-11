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
TString AttachLabel(const TString& in_path,
                    const TString& out_dir,
                    const TString& tree_name="jetAnalyser")
{

    std::cout << "In: " << in_path << std::endl;

    // In
    TFile* in_file = TFile::Open(in_path, "READ");
    TTree* in_tree = dynamic_cast<TTree*>( in_file->Get(tree_name) );

    Int_t kInEntries = in_tree->GetEntries();

    TString in_name = gSystem->BaseName(in_path);

    bool is_qq = in_name.Contains("qq");
    bool is_gg = in_name.Contains("gg");
    bool is_zq = in_name.Contains("zq");

    bool is_quark_jets = is_qq or is_zq;
    bool is_dijet = is_qq or is_gg;

    TString selection = is_dijet ? "balanced" : "pass_Zjets";

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
    TString temp_path = in_path.Copy();
    temp_path.Insert(temp_path.Last('.'), "_TEMPORARY"); 
    TFile* temp_file = TFile::Open(temp_path, "RECREATE");
    TTree* selected_tree = in_tree->CopyTree(selection);
    selected_tree->SetDirectory(temp_file);

    const Int_t kSelectedEntries = selected_tree->GetEntries();

    // variables for checking infnite value.
    Float_t ptD, axis1, axis2;
    Int_t nmult, cmult;
    selected_tree->SetBranchAddress("ptD", &ptD);
    selected_tree->SetBranchAddress("axis1", &axis1);
    selected_tree->SetBranchAddress("axis2", &axis2);
    selected_tree->SetBranchAddress("nmult", &nmult);
    selected_tree->SetBranchAddress("cmult", &cmult);

    TString status = TString::Format(
        "NumGoodJets/NumTotalJets: %d/%d (%.2f) << std::endl",
        kSelectedEntries,
        kInEntries,
        Float_t(kSelectedEntries) / kInEntries);
    std::cout << status << std::endl;


    // Output
    TString out_name = in_name;
    out_name.ReplaceAll("default", selection);
    TString out_path = gSystem->ConcatFileName(out_dir, out_name);

    TFile* out_file = TFile::Open(out_path, "RECREATE");
    TTree* out_tree = selected_tree->CloneTree(0);
    out_tree->SetDirectory(out_file);
    
    Int_t label = is_quark_jets ? 0 : 1;
    
    out_tree->Branch("label", &label, "label/I");
    
    for(Int_t i=0; i < kSelectedEntries; i++){
        selected_tree->GetEntry(i);

        if(IsInfty(cmult)) continue;
        if(IsInfty(nmult)) continue;
        if(IsInfty(ptD)) continue;
        if(IsInfty(axis1)) continue;
        if(IsInfty(axis2)) continue;

        out_tree->Fill();
    }
    
    out_file->Write();
    out_file->Close();

    temp_file->Close();    
    TString rm_temp_cmd = TString::Format("rm %s", temp_path.Data());
    gSystem->Exec(rm_temp_cmd); 

    in_file->Close();


    std::cout << "Out: " << out_path << std::endl << std::endl;
    
    return out_path;
}


TString Step1_AttachLabel(TString in_dir, TString parent_dir)
{
    TString out_dir = gSystem->ConcatFileName(parent_dir, "step1_labeling");
    if(gSystem->AccessPathName(out_dir))
        gSystem->mkdir(out_dir);

    std::vector<TString> paths = ListDir(in_dir, ".root");
    for(auto in_path : paths)
        AttachLabel(in_path, out_dir);

    return out_dir;
}


// Step2
TString Step2_Merge(TString in_dir){

    TString parent_dir = gSystem->DirName(in_dir);
    TString out_dir = gSystem->ConcatFileName(parent_dir, "step2_first_merge");
    if(gSystem->AccessPathName(out_dir))
        gSystem->mkdir(out_dir);

    std::vector< TString > qq, gg, zq, zg;
    qq = ListDir(in_dir, ".root", "qq");
    gg = ListDir(in_dir, ".root", "gg");
    zq = ListDir(in_dir, ".root", "zq");
    zg = ListDir(in_dir, ".root", "zg");

    TString out_fmt = gSystem->ConcatFileName(out_dir, "%s_%d.root");

    // Dijet
    for(auto i=0; i<qq.size(); i++){
        TChain mychain("jetAnalyser");
        mychain.Add(qq[i]);
        mychain.Add(gg[i]);
        TString out_path = TString::Format(out_fmt, "dijet", i); 
        mychain.Merge(out_path);
    }

    // Z+jet
    for(auto i=0; i<zq.size(); i++){
        TChain mychain("jetAnalyser");
        mychain.Add(zq[i]);
        mychain.Add(zg[i]);
        TString out_path = TString::Format(out_fmt, "zjet", i); 
        mychain.Merge(out_path);
    }

    return out_dir;
}


/////////////////////////////////
//      STEP3
////////////////////////////////
TString ShuffleTree(const TString& in_path,
                    const TString& out_dir,
                    const TString& in_numcycle="jetAnalyser")
{

    TFile* in_file = new TFile(in_path, "READ");
    TTree* in_tree = (TTree*) in_file->Get(in_numcycle);
    const Int_t kInEntries = in_tree->GetEntries();

    TString out_filename = gSystem->BaseName(in_path);
    TString out_path = gSystem->ConcatFileName(out_dir, out_filename);

    TFile* out_file = new TFile(out_path, "RECREATE");
    TTree* out_tree = in_tree->CloneTree(0);
    out_tree->SetDirectory(out_file);

    Int_t order[kInEntries];
    std::iota(order, order + kInEntries, 0);
    std::random_shuffle(order, order + kInEntries);


    const Int_t kPrInt_tFreq = static_cast<Int_t>( kInEntries / 10.0 );

    for(int i=0; i < kInEntries; i++){
        in_tree->GetEntry(order[i]);

        if((i % kPrInt_tFreq == 0) or ( (i+1) == kInEntries )){
            std::cout << "(" << 10 * i / kPrInt_tFreq << "%) "
                 << i << "the entires" << std::endl;
        }

        out_tree->Fill();
    }

    out_file->Write();
    out_file->Close();

    in_file->Close();

    return out_path;
}


TString Step3_Shuffle(TString in_dir)
{

    TString parent_dir = gSystem->DirName(in_dir);
    TString out_dir = gSystem->ConcatFileName(parent_dir, "step3_shuffle");
    if(gSystem->AccessPathName(out_dir))
        gSystem->mkdir(out_dir);

    std::vector< TString > dijet_files = ListDir(in_dir, ".root", "dijet");
    std::vector< TString > zjet_files = ListDir(in_dir, ".root", "zjet");

    TString out_path;

    for(auto dj : dijet_files)
        out_path = ShuffleTree(dj, out_dir);
        std::cout << out_path << std::endl << std::endl;

    for(auto zj : zjet_files)
        out_path = ShuffleTree(zj, out_dir);
        std::cout << out_path << std::endl << std::endl;

    return out_dir;
}

//
//  STEP4 Merge
//
TString Step4_Merge(TString in_dir){

    TString parent_dir = gSystem->DirName(in_dir);
    TString out_dir = gSystem->ConcatFileName(parent_dir, "step4_second_merge");
    if(gSystem->AccessPathName(out_dir))
        gSystem->mkdir(out_dir);

    std::vector< TString > dijet_paths, zjet_paths;
    dijet_paths = ListDir(in_dir, ".root", "dijet");
    zjet_paths = ListDir(in_dir, ".root", "zjet");

    // Dijet
    TChain dijet_chain("jetAnalyser");
    TString dijet_chain_path = gSystem->ConcatFileName(out_dir, "dijet.root");
    for(auto dj : dijet_paths)
        dijet_chain.Add(dj);
    dijet_chain.Merge(dijet_chain_path);

    // Z+jet
    TChain zjet_chain("jetAnalyser");
    TString zjet_chain_path = gSystem->ConcatFileName(out_dir, "zjet.root");
    for(auto zj : zjet_paths)
        zjet_chain.Add(zj);
    zjet_chain.Merge(zjet_chain_path);

    return out_dir;
}


TString BalanceClasses(const TString& in_path, const TString& out_dir)
{
    // In Dijet
    std::cout << "In: " << in_path << std::endl;
    TFile* in_file = TFile::Open(in_path, "READ");
    TTree* in_tree = dynamic_cast<TTree*>( in_file->Get("jetAnalyser") );
    const Int_t kInEntries = in_tree->GetEntries();

    Int_t label;
    in_tree->SetBranchAddress("label", &label);

    const Int_t kNumQuark = in_tree->Draw("pt >> tmp_hist_quark", "label == 0", "goff"); 
    const Int_t kNumGluon = in_tree->Draw("pt >> tmp_hist_gluon", "label == 1", "goff"); 

    gDirectory->Delete("tmp_hist_quark");
    gDirectory->Delete("tmp_hist_gluon");

    std::cout << "For the in," << std::endl;
    std::cout << "# of quark jets: " << kNumQuark << std::endl;
    std::cout << "# of gluon jets: " << kNumGluon << std::endl;


    // Output Dijet
    TString out_name = gSystem->BaseName(in_path);
    TString out_path = gSystem->ConcatFileName(out_dir, out_name);

    if ( kNumQuark == kNumGluon ) {
        TString mv_cmd = TString::Format("mv %s %s", in_path.Data(), out_dir.Data());
        gSystem->Exec(mv_cmd);
    }
    else {
        const Int_t kMoreClass = kNumQuark > kNumGluon ? 0 : 1;
        const Int_t kLessNum = kNumQuark > kNumGluon ? kNumGluon : kNumQuark;
        std::cout << "a Class with More examples: " << kMoreClass << std::endl;
        std::cout << "a Number of Less: " << kLessNum << std::endl;

        TFile* out_file = TFile::Open(out_path, "RECREATE");
        TTree* out_tree = in_tree->CloneTree(0);
        out_tree->SetDirectory(out_file);

        Int_t count = 0;
        const Int_t kPrInt_tFreq = kInEntries / 10;
        for(Int_t i=0; i < kInEntries; i++) {
            in_tree->GetEntry(i);

            if( i % kPrInt_tFreq == 0 ) {
                std::cout << "(" << 10 * i / kPrInt_tFreq << "%) "
                          << i << "th entries" << std::endl;
            }

            if( label == kMoreClass )
            {
                if ( count < kLessNum )
                {
                    out_tree->Fill();
                    count++;
                }
                else
                    continue;
            }
            else
            {
                out_tree->Fill();
            }
        }

        const Int_t kNumOutQuark = out_tree->Draw("pt >> tmp_hist_quark", "label == 0", "goff"); 
        const Int_t kNumOutGluon = out_tree->Draw("pt >> tmp_hist_gluon", "label == 1", "goff"); 

        gDirectory->Delete("tmp_hist_quark");
        gDirectory->Delete("tmp_hist_gluon");

        std::cout << std::endl;
        std::cout << "For the out" << std::endl;
        std::cout << "# of quark jets: " << kNumOutQuark << std::endl;
        std::cout << "# of gluon jets: " << kNumOutGluon << std::endl;


        out_file->Write();
        out_file->Close();
    }

    std::cout << "Out: " << out_path << std::endl;
    return out_path;
}



TString Step5_BalanceClasses(const TString& in_dir)
{
    TString out_dir = gSystem->DirName(in_dir);

    TString dijet_path = gSystem->ConcatFileName(in_dir, "dijet.root");
    TString zjet_path = gSystem->ConcatFileName(in_dir, "zjet.root");

    BalanceClasses(dijet_path, out_dir);
    BalanceClasses(zjet_path, out_dir);
    return out_dir; 

}


void macro(const TString& in_dir){
 
    TString parent_dir = gSystem->DirName(in_dir);
    TString merged_dir = gSystem->ConcatFileName(parent_dir, "2-Refined");
    if(gSystem->AccessPathName(merged_dir))
        gSystem->mkdir(merged_dir);
   
    std::cout << "Step 1: Select good jets and attach label" << std::endl << std::endl;
    TString step1_path = Step1_AttachLabel(in_dir, merged_dir);

    std::cout << "Step 1: Merge one quark jet and gluon jet file." << std::endl << std::endl;
    TString step2_path = Step2_Merge(step1_path);

    std::cout << "Step 3: Shuffle small dataset." << std::endl << std::endl;
    TString step3_path = Step3_Shuffle(step2_path);

    std::cout << "Step 4: Merge the entire dataset." << std::endl << std::endl;
    TString step4_path = Step4_Merge(step3_path);
}


Int_t main(Int_t argc, char *argv[])
{
    TString in_dir = argv[1];
    macro(in_dir);
    return 0;
}
