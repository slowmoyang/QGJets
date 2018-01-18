#include "TFile.h"
#include "TTree.h"
#include "TList.h"

#include <iostream>
#include <memory>
#include <algorithm> // std::count_if, std::transform 
#include <numeric> // std::accumulate
#include <functional> // std::plus
#include <tuple> // std::tuple, std::tie, std::make_tuple
#include <vector> // std::vector
#include <map> // std::map
#include <limits> // std::numeric_limits

#include <stdlib.h> // abs

#include "pixelation.cc"

//////////////////////////////////////////////////////
const float kDEtaMax = 0.4;
const float kDPhiMax = 0.4;
//////////////////////////////////////////////////////////
template<typename T>
bool IsInfty(T i) {return std::abs(i) == std::numeric_limits<T>::infinity();}

////////////////////////////////////////////////////////////////////

// macro(name, size);
#define APPLY_ON_IMAGES(macro) \
    macro(image_cpt_33, 33*33); \
    macro(image_npt_33, 33*33); \
    macro(image_cmult_33, 33*33); \
    macro(image_cpt_47, 47*47); \
    macro(image_sqrt_33, 33*33); \
    macro(image_sqrt_47, 47*47); \
    macro(image_sigmoid_33, 33*33); \
    macro(image_sigmoid_47, 47*47); \
    macro(image_electron_33, 33*33); \
    macro(image_muon_33, 33*33); \
    macro(image_photon_33, 33*33); \
    macro(image_chad_33, 33*33); \
    macro(image_nhad_33, 33*33);


TString MakeJetImage(TString const& input_path,
                     TString const& output_dir,
                     TString const& input_key="jetAnalyser"){


    std::cout << "\n#################################################" << std::endl;
    std::cout << "Input: " << input_path << std::endl << std::endl;

    TFile* input_file = TFile::Open(input_path, "READ");
    TTree* input_tree = (TTree*) input_file->Get(input_key);
    const int input_entries = input_tree->GetEntries();
    std::cout << "  Input entries: " << input_entries << std::endl;

    // discriminating variables
    float ptD, axis1, axis2;
    int nmult, cmult;
    // daughters for jet image
    int n_dau;
    std::vector<float> *dau_pt=0, *dau_deta=0, *dau_dphi=0;
    std::vector<int> *dau_charge=0, *dau_pid=0, *dau_ishadronic=0;

    #define SBA(name) input_tree->SetBranchAddress(#name, &name);
    // For image
    SBA(n_dau);
    SBA(dau_pt);
    SBA(dau_deta);
    SBA(dau_dphi);
    SBA(dau_charge);
    SBA(dau_pid);
    SBA(dau_ishadronic);

    // discriminating variables
    SBA(nmult);
    SBA(cmult);
    SBA(ptD);
    SBA(axis1);
    SBA(axis2);

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
        float name[size] = {0.0,}; \
        output_tree->Branch(#name, &name, TString::Format("%s[%d]/F", #name, size));
    APPLY_ON_IMAGES(BRANCH_IMAGE)


    // Pixelation(float eta_up, int eta_num_bins, float phi_up, int phi_num_bins)
    Pixelation pix33 = Pixelation(0.4, 33, 0.4, 33);
    Pixelation pix47 = Pixelation(0.4, 47, 0.4, 47);
    // SqrtPixelation(float eta_up, int eta_num_bins, float phi_up, int phi_num_bins)
    SqrtPixelation sqrt_pix33 = SqrtPixelation(0.4, 33, 0.4, 33);
    SqrtPixelation sqrt_pix47 = SqrtPixelation(0.4, 47, 0.4, 47);
    // SigmoidPixelation(float xup=0.4, int num_pixels=33, flaot threshold=0.1)
    SigmoidPixelation sig_pix33 = SigmoidPixelation(0.4, 33, 0.4, 33, 0.1);
    SigmoidPixelation sig_pix47 = SigmoidPixelation(0.4, 47, 0.4, 47, 0.1);


    #define FILL_ZERO(image, size) \
        std::fill(std::begin(image), std::end(image), 0.0);

    const int print_freq = input_entries / 20;
    // Make image
    float deta, dphi, dth_pt;
    int idx33, idx47, idx_sqrt_33, idx_sqrt_47, idx_sig_33, idx_sig_47; 

    for(unsigned int i=0; i<input_entries; ++i){

    	input_tree->GetEntry(i);

        if(i%print_freq==0)
            cout <<  "(" << 5 * i / print_freq << "%) " << i << "th entries" << std::endl;

        if(IsInfty(cmult)) continue;
        if(IsInfty(nmult)) continue;
        if(IsInfty(ptD)) continue;
        if(IsInfty(axis1)) continue;
        if(IsInfty(axis2)) continue;

        APPLY_ON_IMAGES(FILL_ZERO);

    	for(int d = 0; d < n_dau; ++d){
            deta = dau_deta->at(d);
            dphi = dau_dphi->at(d);

            if((abs(deta) >= kDEtaMax) or (abs(dphi) >= kDPhiMax))
                continue;

            idx33 = pix33.Pixelate(deta, dphi); 
            idx47 = pix47.Pixelate(deta, dphi);
            // Sqrt
            idx_sqrt_33 = sqrt_pix33.Pixelate(deta, dphi);
            idx_sqrt_47 = sqrt_pix47.Pixelate(deta, dphi);
            // Sigmoid
            idx_sig_33 = sig_pix33.Pixelate(deta, dphi);
            idx_sig_47 = sig_pix47.Pixelate(deta, dphi);

            // pT of d-th constituent of a jet
            dth_pt = static_cast<float>(dau_pt->at(d));
            
    	    // charged particle
	        if(dau_charge->at(d)){
	    	    image_cpt_33[idx33] += dth_pt;
        		image_cmult_33[idx33] += 1.0;

	    	    image_cpt_47[idx47] += dth_pt;

                image_sqrt_33[idx_sqrt_33] += dth_pt;
                image_sqrt_47[idx_sqrt_47] += dth_pt;

                image_sigmoid_33[idx_sig_33] += dth_pt;
                image_sigmoid_47[idx_sig_47] += dth_pt;

                // Electron or Positron
                if( abs(dau_pid->at(d)) == 11){
                    image_electron_33[idx33] += dth_pt;
                }
                // Muon or antimuon
                else if( abs( dau_pid->at(d) ) == 13){
                    image_muon_33[idx33] += dth_pt;
                }
                // Charged Hadrons
                else{
                    image_chad_33[idx33] += dth_pt;
                }
	        }
            // Neutral particle
	        else{
        		image_npt_33[idx33] += dth_pt;

                // Neutral Hadron
                if(dau_ishadronic->at(d)) {
                    image_nhad_33[idx33] += dth_pt;
                }
                // Photon
                else {
                    image_photon_33[idx33] += dth_pt;
                }
    	    }

    	}

        output_tree->Fill();
    }

    output_file->Write();
    output_file->Close();
    input_file->Close();


    std::cout << "Output: " << output_path << std::endl;
    return output_path; 
}


std::tuple<TString, TString, TString>
SplitDataset(TString const& input_path, TString const& output_dir)
{
    std::cout << "\n#################################################" << std::endl;
    std::cout << "In: " << input_path << std::endl;

    TFile* input_file = TFile::Open(input_path, "READ");
    TTree* input_tree = (TTree*) input_file->Get("jetAnalyser");
    const int input_entries = input_tree->GetEntries();

    // val = validation
    const int val_start = static_cast<int>(input_entries*0.6);
    const int test_start = static_cast<int>(input_entries*0.8);

    // OUTPUTS
    TString input_dir = gSystem->DirName(input_path);
    TString parent_dir = gSystem->DirName(input_dir);



    TString input_name = gSystem->BaseName(input_path);
    TString name_fmt = input_name.Insert(
        input_name.Last('.'), "%s");

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


    // Sclae training image and fill it. 
    for(unsigned int i=0; i<val_start; i++){
        input_tree->GetEntry(i);
        train_tree->Fill();
    }

    for(unsigned int i=val_start; i<test_start; i++){
        input_tree->GetEntry(i);
        val_tree->Fill();
    }

    for(unsigned int i=test_start; i<input_entries; i++){
        input_tree->GetEntry(i);
        test_tree->Fill();
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

void ScaleImage(float image[], int size, float scale_factor){
    std::transform(image,
                   image+size,
                   image,
                   [scale_factor](float v_)->float{return v_ / scale_factor;});
}

TString PrepTrainingSet(TString const& input_path,
                        TString const& output_dir)
{
    // Input
    TFile* input_file = TFile::Open(input_path, "READ");
    TTree* input_tree = (TTree*)input_file->Get("jetAnalyser");
    const int input_entries = input_tree->GetEntries();

    #define SBA_IMAGE(image, size) \
        float image[size] = {0.}; \
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
    std::map<TString, float> scale_factor;
    std::map<TString, int> num_non_zero;

    for(auto i : image_branches)
    {
        scale_factor[i] = 0.0;
        num_non_zero[i] = 0;
    }

    #define SUM_PIXEL_INTENSITY(image, size) \
        scale_factor[#image] += std::accumulate( \
            image, image + size, 0, std::plus<float>());

    #define COUNT_NON_ZERO(image, size) \
        num_non_zero[#image] += std::count_if( \
            image, image + size, [](float p_){return p_ != 0;});

    for(int i = 0; i < input_entries; i++)
    {
        input_tree->GetEntry(i);
        APPLY_ON_IMAGES(SUM_PIXEL_INTENSITY);
        APPLY_ON_IMAGES(COUNT_NON_ZERO);
    }

    for(TString key : image_branches)
        scale_factor[key] /= num_non_zero[key];


    // Scale training image
    #define SCALE_IMAGE(image, size) ScaleImage(image, size, scale_factor[#image]);

    for(int i=0; i<input_entries; i++){
        input_tree->GetEntry(i);
        APPLY_ON_IMAGES(SCALE_IMAGE);
        output_tree->Fill();
    }


    #define ADD_INFO(image, size) \
        TParameter<float>* param_##image = new TParameter<float>(TString(#image), scale_factor[#image]); \
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
    #define SBAF(name, size) SBA_(float, name, size);

    APPLY_ON_IMAGES(SBAF);

    // 
    TFile* train_file = new TFile(train_path, "READ");
    TTree* train_tree = (TTree*) train_file->Get("jetAnalyser");
    TList* train_info = train_tree->GetUserInfo();


    std::map<TString, float> scale_factor;
    
    TParameter<float>* param;
    TIter iter(train_info);
    while(( param = (TParameter<float>*) iter() )){
        TString name = param->GetName();
        float value = param->GetVal();
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

    const int input_entries = input_tree->GetEntries();
    for(unsigned int i=0; i<input_entries; i++){
        input_tree->GetEntry(i);

        APPLY_ON_IMAGES(SCALE_IMAGE);


        output_tree->Fill();
    }

    output_file->Write();

    std::cout << "Output: " << output_path << "," << std::endl;
    std::cout << "\n#################################################" << std::endl;


    return output_path;
}


void macro(TString input_dir, TString output_dir)
{
    // Input files
    TString dijet_path = gSystem->ConcatFileName(input_dir, "dijet.root");
    TString zjet_path = gSystem->ConcatFileName(input_dir, "zjet.root");

    // Out
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
