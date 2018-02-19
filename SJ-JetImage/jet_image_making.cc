#include "TFile.h"
#include "TTree.h"
#include "TList.h"
#include "TSystem.h"
#include "TParameter.h"
#include "TCollection.h"
#include "TDirectory.h"

#include <iostream>
#include <memory>
#include <algorithm> // std::count_if, std::transform 
#include <numeric> // std::accumulate
#include <functional> // std::plus
#include <tuple> // std::tuple, std::tie, std::make_tuple
#include <vector> // std::vector
#include <map> // std::map
#include <limits> // std::numeric_limits

#include <math.h> // fabs

#include "pixelation.cc"
#include "pdgid.cc"

////////////////////////////////////////////////////////////////////

/*********************************************
 * macro(name, size)
 *
 * > Particle (defulat charged)
 *   - charged = charged particle
 *   - neutral = neutral particle
 *   - chad = charged hadron
 *   - electron = elctron + positron
 *   - muon = muon and anti-muon
 *   - nhad = neutral hadron
 *   - photon
 *   
 * > Variable (defulat pt)
 *   - pt: pT
 *   - mult: multiplicity  
 *
 * 
 * > 
 *   - sigmoid
 *   - sqrt
 *   - sign
 *******************************************/
#define APPLY_ON_IMAGES(macro) \
    macro(image_cpt_33, 33*33); \
    macro(image_npt_33, 33*33); \
    macro(image_cmult_33, 33*33); \
    macro(image_nmult_33, 33*33); \
    macro(image_lepton_33, 33*33); \
    macro(image_electron_33, 33*33); \
    macro(image_muon_33, 33*33); \
    macro(image_photon_33, 33*33); \
    macro(image_chad_33, 33*33); \
    macro(image_nhad_33, 33*33); \
    macro(image_lepton_mult_33, 33*33); \
    macro(image_electron_mult_33, 33*33); \
    macro(image_muon_mult_33, 33*33); \
    macro(image_photon_mult_33, 33*33); \
    macro(image_chad_mult_33, 33*33); \
    macro(image_nhad_mult_33, 33*33); \
    macro(image_cpt_47, 47*47); \
    macro(image_cpt_99, 99*99); \
    macro(image_sqrt_33, 33*33); \
    macro(image_sqrt_47, 47*47); \
    macro(image_sigmoid_33, 33*33); \
    macro(image_sigmoid_47, 47*47); 



TString MakeJetImage(TString const& input_path,
                     TString const& output_dir,
                     TString const& input_key="jetAnalyser"){


    std::cout << "\n#################################################" << std::endl;
    std::cout << "Input: " << input_path << std::endl << std::endl;

    TFile* input_file = TFile::Open(input_path, "READ");
    TTree* input_tree = dynamic_cast<TTree*>( input_file->Get(input_key) );
    const Int_t input_entries = input_tree->GetEntries();
    std::cout << "  Input entries: " << input_entries << std::endl;

    // daughters for jet image
    Int_t n_dau;
    std::vector<Float_t> *dau_pt=0, *dau_deta=0, *dau_dphi=0;
    std::vector<Int_t> *dau_charge=0, *dau_pid=0, *dau_ishadronic=0;

    #define SBA(name) input_tree->SetBranchAddress(#name, &name);
    // For image
    SBA(n_dau);
    SBA(dau_pt);
    SBA(dau_deta);
    SBA(dau_dphi);
    SBA(dau_charge);
    SBA(dau_pid);
    SBA(dau_ishadronic);

    ////////////////////////////////////////////////////////////////////////
    // Output files
    ///////////////////////////////////////////////////////////////////////
    TString input_dir = gSystem->DirName(input_path);
    TString parent_dir = gSystem->DirName(input_dir);

    if(gSystem->AccessPathName(output_dir))
        gSystem->mkdir(output_dir);

    TString output_name = gSystem->BaseName(input_path);
    TString output_path = gSystem->ConcatFileName(output_dir, output_name);

    TFile* output_file = TFile::Open(output_path, "RECREATE");
    TTree* output_tree = input_tree->CloneTree(0);
    output_tree->SetDirectory(output_file);

    // Branch
    #define BRANCH_IMAGE(name, size) \
        Float_t name[size] = {0.0}; \
        output_tree->Branch(#name, &name, TString::Format("%s[%d]/F", #name, size));
    APPLY_ON_IMAGES(BRANCH_IMAGE)


    // Pixelation(Float_t eta_up, Int_t eta_num_bins, Float_t phi_up, Int_t phi_num_bins)
    Pixelation pix33 = Pixelation(0.4, 33, 0.4, 33);
    Pixelation pix47 = Pixelation(0.4, 47, 0.4, 47);
    //SqrtPixelation(Float_t eta_up, Int_t eta_num_bins, Float_t phi_up, Int_t phi_num_bins)
    SqrtPixelation sqrt_pix33 = SqrtPixelation(0.4, 33, 0.4, 33);
    SqrtPixelation sqrt_pix47 = SqrtPixelation(0.4, 47, 0.4, 47);
    // SigmoidPixelation(Float_t xup=0.4, Int_t num_pixels=33, flaot threshold=0.1)
    SigmoidPixelation sig_pix33 = SigmoidPixelation(0.4, 33, 0.4, 33, 0.1);
    SigmoidPixelation sig_pix47 = SigmoidPixelation(0.4, 47, 0.4, 47, 0.1);


    #define FILL_ZERO(image, size) \
        std::fill(image, image + size, 0.0);

    // Make image
    Float_t deta, dphi, daughter_pt;
    Int_t daughter_charge, daughter_pid, daughter_ishadronic;

    Int_t idx33, idx47, idx_sqrt_33, idx_sqrt_47, idx_sig_33, idx_sig_47; 

    const Int_t kPrintFreq = static_cast<Int_t>( input_entries / 10 );

    /******************************
     * 
     *
     * *********************************/
    Int_t num_chad=0, num_electron=0, num_muon=0, num_nhad=0, num_photon=0;

    for(Int_t i = 0; i < input_entries; ++i){
    	input_tree->GetEntry(i);

        if( i % kPrintFreq == 0 ) {
            std::cout << "[" << i / kPrintFreq * 10 << " %]"
                      << i << "th entry" << std::endl; 
        }


        APPLY_ON_IMAGES(FILL_ZERO);

    	for(Int_t d = 0; d < n_dau; d++) {

            deta = dau_deta->at(d);
            dphi = dau_dphi->at(d);

            if( fabs(deta) >= 0.4 ) continue;
            if( fabs(dphi) >= 0.4 ) continue;

            idx33 = pix33.Pixelate(deta, dphi); 

            idx47 = pix47.Pixelate(deta, dphi);
            // Sqrt
            idx_sqrt_33 = sqrt_pix33.Pixelate(deta, dphi);
            idx_sqrt_47 = sqrt_pix47.Pixelate(deta, dphi);
            // Sigmoid
            idx_sig_33 = sig_pix33.Pixelate(deta, dphi);
            idx_sig_47 = sig_pix47.Pixelate(deta, dphi);

            // pT of d-th constituent of a jet
            daughter_pt = dau_pt->at(d);
            daughter_charge = dau_charge->at(d);
            daughter_pid = dau_pid->at(d);
            daughter_ishadronic = dau_ishadronic->at(d);


    	    // charged particle
	        if(daughter_charge) { 
	    	    image_cpt_33[idx33] += daughter_pt;
        		image_cmult_33[idx33] += 1.0;

	    	    image_cpt_47[idx47] += daughter_pt;

                image_sqrt_33[idx_sqrt_33] += daughter_pt;
                image_sqrt_47[idx_sqrt_47] += daughter_pt;

                image_sigmoid_33[idx_sig_33] += daughter_pt;
                image_sigmoid_47[idx_sig_47] += daughter_pt;


                // Electron or Positron
                if( abs( daughter_pid ) == PdgId::kElectron ){
                    image_electron_33[idx33] += daughter_pt;
                    image_electron_mult_33[idx33] += 1.0;

                    image_lepton_33[idx33] += daughter_pt;
                    image_lepton_mult_33[idx33] += 1.0;

                    num_electron++;
                }
                // Muon or antimuon
                else if( abs( daughter_pid ) == PdgId::kMuon ){
                    image_muon_33[idx33] += daughter_pt;
                    image_muon_mult_33[idx33] += 1.0;

                    image_lepton_33[idx33] += daughter_pt;
                    image_lepton_33[idx33] += 1.0;

                    num_muon++;
                }
                // Charged Hadrons
                else{
                    image_chad_33[idx33] += daughter_pt;
                    image_chad_mult_33[idx33] += 1.0;

                    num_chad++;
                }
	        }
            // Neutral particle
	        else{
        		image_npt_33[idx33] += daughter_pt;
        		image_nmult_33[idx33] += 1.0;

                // Neutral Hadron
                if( daughter_ishadronic ) {
                    image_nhad_33[idx33] += daughter_pt;
                    image_nhad_mult_33[idx33] += 1.0;

                    num_nhad++;
                }
                // Photon
                else {
                    image_photon_33[idx33] += daughter_pt;
                    image_photon_mult_33[idx33] += 1.0;

                    num_photon++;
                }
    	    }

    	}

        output_tree->Fill();
    }

    output_file->Write();
    output_file->Close();
    input_file->Close();

    std::cout << "# of charged hadron: " << num_chad << std::endl;
    std::cout << "# of electron:       " << num_electron << std::endl;
    std::cout << "# of muon:           " << num_muon << std::endl;
    std::cout << "# of netural hadron: " << num_nhad << std::endl;
    std::cout << "# of photon:         " << num_photon << std::endl;


    std::cout << "Output: " << output_path << std::endl;
    return output_path; 
}


std::tuple<TString, TString, TString>
SplitDataset(TString const& input_path,
             TString const& output_dir)
{
    std::cout << "\n#################################################" << std::endl;
    std::cout << "In: " << input_path << std::endl;

    TFile* input_file = TFile::Open(input_path, "READ");
    TTree* input_tree = dynamic_cast<TTree*>( input_file->Get("jetAnalyser") );
    const Int_t input_entries = input_tree->GetEntries();

    Int_t label;
    input_tree->SetBranchAddress("label", &label);

    // OUTPUTS
    TString input_dir = gSystem->DirName(input_path);
    TString parent_dir = gSystem->DirName(input_dir);

    TString input_name = gSystem->BaseName(input_path);
    TString name_fmt = input_name.Insert(input_name.Last('.'), "%s");

    // e.g. dijet_training_set.root
    TString train_name = TString::Format(name_fmt, "_training_set");
    TString val_name = TString::Format(name_fmt, "_validation_set");
    TString test_name = TString::Format(name_fmt, "_test_set");

    TString train_path = gSystem->ConcatFileName(output_dir, train_name);
    TString val_path = gSystem->ConcatFileName(output_dir, val_name);
    TString test_path = gSystem->ConcatFileName(output_dir, test_name);

    // training
    TFile* train_file = TFile::Open(train_path, "RECREATE");
    TTree* train_tree = input_tree->CloneTree(0);
    train_tree->SetDirectory(train_file);

    // vaildiation
    TFile* val_file = TFile::Open(val_path, "RECREATE");
    TTree* val_tree = input_tree->CloneTree(0);
    val_tree->SetDirectory(val_file);

    // test
    TFile* test_file = TFile::Open(test_path, "RECREATE");
    TTree* test_tree = input_tree->CloneTree(0);
    test_tree->SetDirectory(test_file);


    const Int_t kNumQuark = input_tree->Draw("pt >> tmp_hist_quark", "label == 0", "goff"); 
    const Int_t kNumGluon = input_tree->Draw("pt >> tmp_hist_gluon", "label == 1", "goff"); 
    gDirectory->Delete("tmp_hist_quark");
    gDirectory->Delete("tmp_hist_gluon");

    std::cout << "# of jets: " << input_entries << std::endl;
    std::cout << "# of Quark jets: " << kNumQuark << std::endl;
    std::cout << "# of Gluon jets: " << kNumGluon << std::endl;

    Int_t quark_count = 0;
    const Int_t kQuarkValStart = static_cast<Int_t>(kNumQuark*0.6);
    const Int_t kQuarkTestStart = static_cast<Int_t>(kNumQuark*0.8);

    std::cout << "kQuarkValStart: " << kQuarkValStart << std::endl;
    std::cout << "kQuarkTestStart: " << kQuarkTestStart << std::endl;

    Int_t gluon_count = 0;
    const Int_t kGluonValStart = static_cast<Int_t>(kNumGluon*0.6);
    const Int_t kGluonTestStart = static_cast<Int_t>(kNumGluon*0.8);

    std::cout << "kGluonValStart: " << kGluonValStart << std::endl;
    std::cout << "kGluonTestStart: " << kGluonTestStart << std::endl;

    const Int_t kPrintFreq = static_cast<Int_t>( input_entries / 10 );

    for(Int_t i=0; i < input_entries; i++){
        input_tree->GetEntry(i);

        if( i % kPrintFreq == 0 ) {
            std::cout << "[" << i / kPrintFreq * 10 << " %]"
                      << i << "th entry" << std::endl; 
        }


        // quark jet
        if( label == 0 ){

            if(quark_count < kQuarkValStart) {
                train_tree->Fill();
            }
            else if ( quark_count < kQuarkTestStart ) {
                val_tree->Fill();
            }
            else {
                test_tree->Fill();
            }

            quark_count++;
        }
        else if ( label == 1 ){

            if( gluon_count < kGluonValStart ) {
                train_tree->Fill();
            }
            else if ( gluon_count < kGluonTestStart ) {
                val_tree->Fill();
            }
            else {
                test_tree->Fill();
            }

            gluon_count++;
        }
        else{
            std::cout << ":p"  << std::endl;
        }
    }

    train_file->Write();
    val_file->Write();
    test_file->Write();

    train_file->Close();
    val_file->Close();
    test_file->Close();

    input_file->Close();

    std::cout << "Out: " << train_path << "," << std::endl;
    std::cout << "        " << val_path << "," << std::endl;
    std::cout << "        " << test_path << std::endl;
    std::cout << "\n#################################################" << std::endl;

    return std::make_tuple(train_path, val_path, test_path);
}

inline void ScaleImage(Float_t image[], Int_t size, Float_t scale_factor){
    std::transform(image,
                   image+size,
                   image,
                   [scale_factor](Float_t v_)->Float_t{return v_ / scale_factor;});
}

TString PrepTrainingSet(TString const& input_path,
                        TString const& output_dir)
{
    // Input
    TFile* input_file = TFile::Open(input_path, "READ");
    TTree* input_tree = (TTree*)input_file->Get("jetAnalyser");
    const Int_t input_entries = input_tree->GetEntries();

    #define SBA_IMAGE(image, size) \
        Float_t image[size] = {0.}; \
        input_tree->SetBranchAddress(#image, &image);
    APPLY_ON_IMAGES(SBA_IMAGE);

    // Out
    TString output_name = gSystem->BaseName(input_path);
    TString output_path = gSystem->ConcatFileName(output_dir, output_name);

    TFile* output_file = TFile::Open(output_path, "RECREATE");
    TTree* output_tree = input_tree->CloneTree(0);  
    output_tree->SetDirectory(output_file);

    // Extract..
    std::vector<TString> image_branches;
    TString branch_name;
    for(auto i : *input_tree->GetListOfBranches()){
        branch_name = i->GetName();
        if(branch_name.Contains("image"))
            image_branches.push_back(branch_name);
    }

    // Calculate scale factor;
    std::map<TString, Float_t> scale_factor;
    std::map<TString, Int_t> num_non_zero;

    for(auto i : image_branches)
    {
        scale_factor[i] = 0.0;
        num_non_zero[i] = 0;
    }

    #define SUM_PIXEL_INTENSITY(image, size) \
        scale_factor[#image] += std::accumulate( \
            image, image + size, 0, std::plus<Float_t>());

    #define COUNT_NON_ZERO(image, size) \
        num_non_zero[#image] += std::count_if( \
            image, image + size, [](Float_t p_){return p_ != 0;});

    for(Int_t i = 0; i < input_entries; i++)
    {
        input_tree->GetEntry(i);
        APPLY_ON_IMAGES(SUM_PIXEL_INTENSITY);
        APPLY_ON_IMAGES(COUNT_NON_ZERO);
    }

    for(TString key : image_branches)
        scale_factor[key] /= num_non_zero[key];


    // Scale training image
    #define SCALE_IMAGE(image, size) ScaleImage(image, size, scale_factor[#image]);

    for(Int_t i=0; i<input_entries; i++){
        input_tree->GetEntry(i);
        APPLY_ON_IMAGES(SCALE_IMAGE);
        output_tree->Fill();
    }


    #define ADD_INFO(image, size) \
        TParameter<Float_t>* param_##image = new TParameter<Float_t>(TString(#image), scale_factor[#image]); \
        output_tree->GetUserInfo()->Add(param_##image);

    APPLY_ON_IMAGES(ADD_INFO);

    output_file->Write();
    std::cout << "Out: " << output_path << std::endl;

    return output_path;
}


TString PrepTestSet(TString const& input_path,
                    TString const& train_path){

    // File to be preprocessed
    std::cout << "\n#################################################" << std::endl;
    std::cout << "A File to be preprocessed: " << input_path << std::endl;
    std::cout << "A File having scale facotr: " << train_path << std::endl;

    TFile* input_file = new TFile(input_path, "READ");
    TTree* input_tree = (TTree*) input_file->Get("jetAnalyser");

    #define SBA_(type, name, size) \
        type name[size] = {0.}; input_tree->SetBranchAddress(#name, &name);
    #define SBAF(name, size) SBA_(Float_t, name, size);

    APPLY_ON_IMAGES(SBAF);

    // 
    TFile* train_file = new TFile(train_path, "READ");
    TTree* train_tree = (TTree*) train_file->Get("jetAnalyser");
    TList* train_info = train_tree->GetUserInfo();


    std::map<TString, Float_t> scale_factor;
    
    TParameter<Float_t>* param;
    TIter iter(train_info);
    while(( param = (TParameter<Float_t>*) iter() )){
        TString name = param->GetName();
        Float_t value = param->GetVal();
        scale_factor[name] = value;
    }
   

    // OUTPUTS
    TString output_dir = gSystem->DirName(train_path);

    TString output_name = gSystem->BaseName(input_path);

    TString output_path = gSystem->ConcatFileName(output_dir, output_name);

    TFile* output_file = new TFile(output_path, "RECREATE");
    TTree* output_tree = input_tree->CloneTree(0);
    output_tree->SetDirectory(output_file);

    #define SCALE_IMAGE(image, size) \
        ScaleImage(image, size, scale_factor[#image]);

    const Int_t input_entries = input_tree->GetEntries();

    for(Int_t i=0; i<input_entries; i++){
        input_tree->GetEntry(i);

        APPLY_ON_IMAGES(SCALE_IMAGE);


        output_tree->Fill();
    }

    output_file->Write();

    std::cout << "Output: " << output_path << "," << std::endl;
    std::cout << "\n#################################################" << std::endl;


    return output_path;
}


void macro(const TString& input_dir){
    // Input files
    TString dijet_path = gSystem->ConcatFileName(input_dir, "dijet.root");
    TString zjet_path = gSystem->ConcatFileName(input_dir, "zjet.root");

    // Output directory
    TString parent_dir = gSystem->DirName(input_dir);
    TString output_dir = gSystem->ConcatFileName(parent_dir, "3-JetImage");
    //TString output_dir = "./test";

    if(gSystem->AccessPathName(output_dir))
        gSystem->mkdir(output_dir);

    // Step1: MakeJetImage
    TString dijet_image_path, zjet_image_path;
    dijet_image_path = MakeJetImage(dijet_path, output_dir);
    zjet_image_path = MakeJetImage(zjet_path, output_dir);

    // Step2: SplitDataset
    TString dijet_train, dijet_val, dijet_test, zjet_train, zjet_val, zjet_test;
    std::tie(dijet_train, dijet_val, dijet_test) = SplitDataset(dijet_image_path, output_dir);
    std::tie(zjet_train, zjet_val, zjet_test) = SplitDataset(zjet_image_path, output_dir);


    // -------------------
    TString dijet_set_dir = gSystem->ConcatFileName(output_dir, "dijet_set");
    TString zjet_set_dir = gSystem->ConcatFileName(output_dir, "zjet_set");
    for(auto out_dir : {dijet_set_dir, zjet_set_dir}){
        if(gSystem->AccessPathName(out_dir))
            gSystem->mkdir(out_dir);
    }

    // Step3: PrepTrainingSet
    TString dijet_train_prep, zjet_train_prep;
    dijet_train_prep = PrepTrainingSet(dijet_train, dijet_set_dir);
    zjet_train_prep = PrepTrainingSet(zjet_train, zjet_set_dir);

    // Step4: PrepTestSet
    PrepTestSet(dijet_val, dijet_train_prep);
    PrepTestSet(dijet_test, dijet_train_prep);
    PrepTestSet(dijet_val, zjet_train_prep);
    PrepTestSet(dijet_test, zjet_train_prep);

    PrepTestSet(zjet_val, dijet_train_prep);
    PrepTestSet(zjet_test, dijet_train_prep);
    PrepTestSet(zjet_val, zjet_train_prep);
    PrepTestSet(zjet_test, zjet_train_prep);
}

int  main(int argc, char *argv[]){
    TString input_dir = argv[1];
    macro(input_dir);

    return 0;
}
