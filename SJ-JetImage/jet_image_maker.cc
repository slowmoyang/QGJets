<<<<<<< HEAD
#include "../../External/jsoncpp/jsoncpp.cpp"
=======
#include "TString.h"
#include "TSystem.h"
#include "TTree.h"
#include "TFile.h"
#include "TVectorD.h"

#include<iostream>
#include<fstream>

using std::cout;
using std::endl;
>>>>>>> upstream/master

//////////////////////////////////////////////////////
const int kChannel = 3;
const int kHeight = 33;
const int kWidth = 33;
const int kPixelsPerChannel = kHeight * kWidth;
const int kTotalPixels = kChannel * kPixelsPerChannel;

const float kDEtaMax = 0.4;
const float kDPhiMax = 0.4;

const TString kTreeName = "jet";

<<<<<<< HEAD
const TString kOutputDir = "$JET/Data/image33x33";
=======
const TString kOutputDir = "image33x33";
>>>>>>> upstream/master
//////////////////////////////////////////////////////////

bool IsNotZero(int i) {return i!=0;}

template<typename T>
bool IsInfty(T i) {return std::abs(i) == std::numeric_limits<T>::infinity();}

////////////////////////////////////////////////////////////////////

TString MakeDataset(TString const& input_path,
<<<<<<< HEAD
                    TString const& input_key="jetAnalyser"){

=======
                    TString const& input_key="jetAnalyser")
{
>>>>>>> upstream/master
    std::cout << "\n#################################################" << endl;
    std::cout << "Input: " << input_path << endl;

    TFile* input_file = new TFile(input_path, "READ");
    TTree* input_tree = (TTree*) input_file->Get(input_key);
    const int kInputEntries = input_tree->GetEntries();

    // discriminating variables
    float ptD, axis1, axis2;
    int nmult, cmult;
    // daughters for jet image
    int n_dau;
    std::vector<float> *dau_pt=0, *dau_deta=0, *dau_dphi=0;
    std::vector<int> *dau_charge=0;
    // label
    int label[2];
    // additional information
    float pt, eta;

    #define SBA(name) input_tree->SetBranchAddress(#name, &name);
    // image
    SBA(n_dau);
    SBA(dau_pt);
    SBA(dau_deta);
    SBA(dau_dphi);
    SBA(dau_charge);
    // discriminating variables
    SBA(nmult);
    SBA(cmult);
    SBA(ptD);
    SBA(axis1);
    SBA(axis2);
    // label
    SBA(label);
    // additional information
    SBA(pt);
    SBA(eta);

    ////////////////////////////////////////////////////////////////////////
    // Output files
    ///////////////////////////////////////////////////////////////////////
    TString output_name = gSystem->BaseName(input_path);
    TString output_path = gSystem->ConcatFileName(kOutputDir, output_name);

    TFile* output_file = new TFile(output_path, "RECREATE");
    // TTree(name, title)
    TTree* output_tree = new TTree(kTreeName, kTreeName);
    output_tree->SetDirectory(output_file);

    float image[kTotalPixels];
    float variables[5];

    TString image_bufsize = TString::Format("image[%d]/F", kTotalPixels);

    output_tree->Branch("image",           &image,           image_bufsize);
    output_tree->Branch("variables",       &variables,       "variables[5]/F");
    output_tree->Branch("label",           &label,           "label[2]/I");
    output_tree->Branch("pt",              &pt,              "pt/F");
    output_tree->Branch("eta",             &eta,             "eta/F");

    for(unsigned int i=0; i<kInputEntries; ++i){
    	input_tree->GetEntry(i);
        if(IsInfty(cmult)) continue;
        if(IsInfty(nmult)) continue;
        if(IsInfty(ptD)) continue;
        if(IsInfty(axis1)) continue;
        if(IsInfty(axis2)) continue;

    	//Variables
    	variables[0] = cmult;
    	variables[1] = nmult;
    	variables[2] = ptD;
    	variables[3] = axis1;
    	variables[4] = axis2;
	
    	// Init imge array
    	std::fill(std::begin(image), std::end(image), 0.0);

        // Make image
    	for(int d = 0; d < n_dau; ++d){
            // width in [0, kWidth)
    	    int w = int((dau_deta->at(d) + kDEtaMax) / (2*kDEtaMax/kWidth));
            // height in [0, kHeight)
    	    int h = int((dau_dphi->at(d) + kDPhiMax) / (2*kDPhiMax/kHeight));

    	    if((w<0)or(w>=kWidth)or(h<0)or(h>=kHeight))
                continue;

    	    // charged particle
	        if(dau_charge->at(d)){
    	    	// pT
	    	    image[kWidth*h + w] += float(dau_pt->at(d));
                // multiplicity
        		image[2*kHeight*kWidth + kWidth*h + w] += 1.0;
	        }
            // neutral particle
	        else{
        		image[kHeight*kWidth + kWidth*h + w] += float(dau_pt->at(d));
    	    }

    	}

        output_tree->Fill();
    }

    output_file->Write();
    output_file->Close();
    input_file->Close();


    std::cout << "Output: " << output_path << endl;
    return output_path; 
}


/*
Preprocess trainin dataset
  - Scaling
*/
<<<<<<< HEAD
std::tuple<TString, TString> PrepTrainData(TString const& input_path){

=======
TString PrepTrainData(TString const& input_path)
{
>>>>>>> upstream/master
    TFile* input_file = TFile::Open(input_path);
    TTree* input_tree = (TTree*) input_file->Get(kTreeName);

    TString input_name = gSystem->BaseName(input_path);
    const int kInputEntries = input_tree->GetEntries();

    float image[kTotalPixels], variables[5], pt, eta;
    int label[2];
    #define SBA(name) input_tree->SetBranchAddress(#name, &name);
    SBA(image);
    SBA(variables);
    SBA(pt);
    SBA(eta);
    SBA(label);

    // Calculate the mean value of each discriminating variables;
    float var_mean[5];

    //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    // 0:cpt, 1:npt, 2: cmu
    float pixel_sum[kChannel] = {0.};
    int pixel_count[kChannel] = {0};
    for(unsigned int i=0; i<kInputEntries; ++i){
        input_tree->GetEntry(i);

        // variables
        for(auto j : {0, 1, 2, 3, 4})
            var_mean[j] += variables[j];

        // image
        for(int j=0; j < kChannel; j++){
            pixel_sum[j] += std::accumulate(
                image + j*kPixelsPerChannel, // first
                image + (j+1)*kPixelsPerChannel, //last: the range of elements to sum
                float(0), // init: initial value of the sum
                std::plus<float>()); // op: binary operation function

            pixel_count[j] += std::count_if(
                image + j*kPixelsPerChannel,
                image + (j+1)*kPixelsPerChannel,
                IsNotZero);
        }
    }

    std::transform(std::begin(var_mean),
                   std::end(var_mean),
                   var_mean, 
                   [kInputEntries](float v_)->float {return v_ / kInputEntries;});

    float pixel_mean[kChannel];
    for(int c=0; c < kChannel; c++)
        pixel_mean[c] = pixel_sum[c] / pixel_count[c];

<<<<<<< HEAD

    TString scale_para_name = input_name.Contains("dijet") ? "scale_para_dijet.json" : "scale_para_zjet.json";
    TString scale_para_path = gSystem->ConcatFileName(kOutputDir, scale_para_name);
    std::ofstream scale_para_file(scale_para_path);

    Json::Value scale_para;
    // variables
    scale_para["variables"]["cmult"] = var_mean[0];
    scale_para["variables"]["nmult"] = var_mean[1];
    scale_para["variables"]["ptD"]   = var_mean[2];
    scale_para["variables"]["axis1"] = var_mean[3];
    scale_para["variables"]["axis2"] = var_mean[4];
    // image
    scale_para["image"]["cpt"] = pixel_mean[0];
    scale_para["image"]["npt"] = pixel_mean[1];
    scale_para["image"]["cmu"] = pixel_mean[2];

    Json::StyledWriter styled_writer;
    scale_para_file << styled_writer.write(scale_para);
    scale_para_file.close();


=======
>>>>>>> upstream/master
    // Name output file
    TString output_name = input_name.Insert(input_name.Last('.'), "_prep");
    TString output_path = gSystem->ConcatFileName(kOutputDir, output_name);

    TFile* output_file = new TFile(output_path, "RECREATE");
<<<<<<< HEAD
=======
    output_file->mkdir("variables");
    output_file->mkdir("images");
    // variables
    output_file->Cd("/variables");
    TVectorD cmult(1), nmult(1), ptD(1), axis1(1), axis2(1);
    cmult[0] = var_mean[0]; cmult.Write("cmult");
    nmult[0] = var_mean[1]; nmult.Write("nmult");
    ptD[0]   = var_mean[2]; ptD.Write("ptD");
    axis1[0] = var_mean[3]; axis1.Write("axis1");
    axis2[0] = var_mean[4]; axis2.Write("axis2");
    // image
    output_file->Cd("/images");
    TVectorD cpt(1), npt(1), cmu(1);
    cpt[0] = pixel_mean[0]; cpt.Write("cpt");
    npt[0] = pixel_mean[1]; npt.Write("npt");
    cmu[0] = pixel_mean[2]; cmu.Write("cmu");
    output_file->Cd("/");

>>>>>>> upstream/master
    TTree* output_tree = (TTree*) input_tree->CloneTree(0);
    output_tree->SetDirectory(output_file);

    const int kPrintFreq = int(kInputEntries/float(10));
    for(int i=0; i<kInputEntries; ++i){
        input_tree->GetEntry(i);

        if(i%kPrintFreq==0){
            cout <<  "(" << 10 * i / kPrintFreq << "%) "
                 << i << "th entries" << endl;
        }

        for(int j=0; j<5; ++j)
            variables[j] /= var_mean[j];

        for(auto j=0; j < kPixelsPerChannel; ++j)
            for(int c=0; c < kChannel; c++)
                image[j + c*kPixelsPerChannel] /= pixel_mean[j];

        output_tree->SetDirectory(output_file);
        output_tree->Fill();
    }

    output_tree->Print();
<<<<<<< HEAD
=======

>>>>>>> upstream/master
    output_file->Write();
    output_file->Close();

    input_file->Close();

<<<<<<< HEAD
    return std::make_tuple(output_path, scale_para_path);
=======
    return output_path;
>>>>>>> upstream/master
}


TString PrepTestData(TString const& input_path,
                     TString const& scale_para_path){

    TFile* input_file = new TFile(input_path, "READ");
    TTree* input_tree = (TTree*) input_file->Get(kTreeName);

    TString input_name = gSystem->BaseName(input_path);
    const int kInputEntries = input_tree->GetEntries();

    float image[kTotalPixels], variables[5], pt, eta;
    int label[2];
    #define SBA(name) input_tree->SetBranchAddress(#name, &name);
    SBA(image);
    SBA(variables);
    SBA(pt);
    SBA(eta);
    SBA(label);

    // Read scale parameter from json file.
<<<<<<< HEAD
    std::ifstream json_file(scale_para_path);

    Json::Value json_root;
    Json::Reader json_reader;

    if(not json_reader.parse(json_file, json_root, true)){
        std::cout << "Failed to parse configuration" << endl
                 << json_reader.getFormattedErrorMessages();
    }

    float var_mean[5];
    var_mean[0] = json_root["variables"]["cmult"].asFloat();
    var_mean[1] = json_root["variables"]["nmult"].asFloat();
    var_mean[2] = json_root["variables"]["ptD"].asFloat();
    var_mean[3] = json_root["variables"]["axis1"].asFloat();
    var_mean[4] = json_root["variables"]["axis2"].asFloat();

    float pixel_mean[3];
    pixel_mean[0] = json_root["image"]["cpt"].asFloat(); 
    pixel_mean[1] = json_root["image"]["npt"].asFloat(); 
    pixel_mean[2] = json_root["image"]["cmu"].asFloat(); 
=======
    TFile* scale_file = new TFile(scale_para_path, "READ");

    float var_mean[5];
    var_mean[0] = (* (TVectorD*) scale_file->Get("/variables/cmult"))[0];
    var_mean[1] = (* (TVectorD*) scale_file->Get("/variables/nmult"))[0];
    var_mean[2] = (* (TVectorD*) scale_file->Get("/variables/ptD"))[0];
    var_mean[3] = (* (TVectorD*) scale_file->Get("/variables/axis1"))[0];
    var_mean[4] = (* (TVectorD*) scale_file->Get("/variables/axis2"))[0];

    float pixel_mean[3];
    pixel_mean[0] = (* (TVectorD*) scale_file->Get("/images/cpt"))[0]; 
    pixel_mean[1] = (* (TVectorD*) scale_file->Get("/images/npt"))[0]; 
    pixel_mean[2] = (* (TVectorD*) scale_file->Get("/images/cmu"))[0]; 
>>>>>>> upstream/master


    TString suffix = scale_para_path.Contains("dijet") ? "_after_dijet" : "_after_zjet";

    TString output_name = input_name.Insert(input_name.Last('.'), suffix);
    TString output_path = gSystem->ConcatFileName(kOutputDir, output_name);

    TFile* output_file = new TFile(output_path, "RECREATE");
    TTree* output_tree = (TTree*) input_tree->CloneTree(0);
    output_tree->SetDirectory(output_file);


    const int kPrintFreq = int(kInputEntries/float(10));
    for(int i=0; i<kInputEntries; ++i){
        input_tree->GetEntry(i);

        if(i%kPrintFreq==0){
            cout <<  "(" << 10 * i / kPrintFreq << "%) "
                 << i << "th entries" << endl;
        }

        for(int j=0; j<5; ++j)
            variables[j] /= var_mean[j];

        for(auto j=0; j < kPixelsPerChannel; ++j)
            for(int c = 0; c < kChannel; c++)
                image[j + c*kPixelsPerChannel] /= pixel_mean[c];

        output_tree->SetDirectory(output_file);
        output_tree->Fill();
    }

    output_tree->Print();
    output_file->Write();
    output_file->Close();

    input_file->Close();

    return output_path;
}

<<<<<<< HEAD

void macro(){

    TString input_fmt = "$JET/Data/FastSim_pt_100_500/%s";
    TString dijet_train = TString::Format(input_fmt, "dijet_train.root");
    TString dijet_test = TString::Format(input_fmt, "dijet_test.root");
    TString zjet_train = TString::Format(input_fmt, "zjet_train.root");
    TString zjet_test = TString::Format(input_fmt, "zjet_test.root");
=======
void macro();
int main(int argc, char *argv[])
{
  macro();
}

void macro()
{

    TString input_fmt = "step5/%s";
    TString dijet_train = TString::Format(input_fmt, "dijet_train.root");
    TString dijet_test = TString::Format(input_fmt, "dijet_test.root");
    TString zjet_train = TString::Format(input_fmt, "z_jet_train.root");
    TString zjet_test = TString::Format(input_fmt, "z_jet_test.root");
>>>>>>> upstream/master

    if(gSystem->AccessPathName(kOutputDir))
        gSystem->mkdir(kOutputDir);

<<<<<<< HEAD
    // Make dataset
=======
    std::cout <<  "Make dataset" << std::endl;
>>>>>>> upstream/master
    TString dj_train = MakeDataset(dijet_train);
    TString dj_test = MakeDataset(dijet_test);
    TString zj_train = MakeDataset(zjet_train);
    TString zj_test = MakeDataset(zjet_test);

<<<<<<< HEAD
    // Preprocess training dataset
    TString dj_train_prep, dj_scale_para;
    std::tie(dj_train_prep, dj_scale_para) = PrepTrainData(dj_train);

    TString zj_train_prep, zj_scale_para;
    std::tie(zj_train_prep, zj_scale_para) = PrepTrainData(zj_train);

    // Preprocess test dataset
    // Dijet test dataset for classifiers trained on Dijet dataset
    PrepTestData(dj_test, dj_scale_para); 
    // Dijet test dataset for classifiers trained on Z+jet dataset
    PrepTestData(dj_test, zj_scale_para);
    // Z+jet test dataset for classifiers trained on Dijet dataset
    PrepTestData(zj_test, dj_scale_para);
    // Z+jet test dataset for classifiers trained on Z+jet dataset
    PrepTestData(zj_test, zj_scale_para);

}



=======
    std::cout <<  "Preprocess training dataset" << std::endl;
    TString dj_train_prep = PrepTrainData(dj_train);
    TString zj_train_prep = PrepTrainData(zj_train);
    
    std::cout <<  "Preprocess test dataset" << std::endl;
    // Dijet test dataset for classifiers trained on Dijet dataset
    PrepTestData(dj_test, TString::Format("%s/%s_prep.root", kOutputDir.Data(), "dijet_train"));
    // Dijet test dataset for classifiers trained on Z+jet dataset
    PrepTestData(dj_test, TString::Format("%s/%s_prep.root", kOutputDir.Data(), "z_jet_train"));
    // Z+jet test dataset for classifiers trained on Dijet dataset
    PrepTestData(zj_test, TString::Format("%s/%s_prep.root", kOutputDir.Data(), "dijet_train"));
    // Z+jet test dataset for classifiers trained on Z+jet dataset
    PrepTestData(zj_test, TString::Format("%s/%s_prep.root", kOutputDir.Data(), "z_jet_train"));

}
>>>>>>> upstream/master
