#include "jsoncpp/jsoncpp.cpp"

//////////////////////////////////////////////////////
const float kDEtaMax = 0.4;
const float kDPhiMax = 0.4;

const TString kTreeName = "jetAnalyser";
//////////////////////////////////////////////////////////

bool IsNotZero(int i) {return i!=0;}

template<typename T>
bool IsInfty(T i) {return std::abs(i) == std::numeric_limits<T>::infinity();}


std::tuple<int, int> CalcIndices(float x, float y,
                                 float x_size=33, float y_size=33,
                                 float x_max=0.4, float y_max=0.4)
{
    int x_idx = int((x+x_max)/(2*x_max/x_size));
    int y_idx = int((y+y_max)/(2*y_max/y_size));
    return std::make_tuple(x_idx, y_idx);
}



////////////////////////////////////////////////////////////////////

TString MakeJetImage(TString const& input_path,
                     TString const& input_key="jetAnalyser"){

    std::cout << "\n#################################################" << endl;
    std::cout << "Input: " << input_path << endl;

    TFile* input_file = new TFile(input_path, "READ");
    TTree* input_tree = (TTree*) input_file->Get(input_key);
    const int input_entries = input_tree->GetEntries();

    // discriminating variables
    float ptD, axis1, axis2;
    int nmult, cmult;
    // daughters for jet image
    int n_dau;
    std::vector<float> *dau_pt=0, *dau_deta=0, *dau_dphi=0;
    std::vector<int> *dau_charge=0;

    #define SBA(name) input_tree->SetBranchAddress(#name, &name);
    // For image
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

    ////////////////////////////////////////////////////////////////////////
    // Output files
    ///////////////////////////////////////////////////////////////////////
    TString input_dir = gSystem->DirName(input_path);
    TString parent_dir = gSystem->DirName(input_dir);

    TString output_dir = gSystem->ConcatFileName(parent_dir, "jet_image");
    if(gSystem->AccessPathName(output_dir))
        gSystem->mkdir(output_dir);

    TString output_name = gSystem->BaseName(input_path);
    TString output_path = gSystem->ConcatFileName(output_dir, output_name);

    TFile* output_file = new TFile(output_path, "RECREATE");
    TTree* output_tree = input_tree->CloneTree(0);
    output_tree->SetDirectory(output_file);

    #define BRANCH_IMAGE(name, num_pixels) \
        float name[num_pixels]; output_tree->Branch(#name, &name, TString::Format("%s[%d]/F", #name, num_pixels));

    BRANCH_IMAGE(image_cpt33, 33*33);
    BRANCH_IMAGE(image_npt33, 33*33);
    BRANCH_IMAGE(image_cmult33, 33*33);

    BRANCH_IMAGE(image_cpt47, 47*47);

    const int print_freq = int(input_entries/float(20));
    for(unsigned int i=0; i<input_entries; ++i){

    	input_tree->GetEntry(i);

        if(i%print_freq==0)
            cout <<  "(" << 20 * i / print_freq << "%) " << i << "th entries" << endl;

        if(IsInfty(cmult)) continue;
        if(IsInfty(nmult)) continue;
        if(IsInfty(ptD)) continue;
        if(IsInfty(axis1)) continue;
        if(IsInfty(axis2)) continue;

    	// Init imge array
    	std::fill(std::begin(image_cpt33),   std::end(image_cpt33),   0.0);
    	std::fill(std::begin(image_cmult33), std::end(image_cmult33), 0.0);
    	std::fill(std::begin(image_npt33),   std::end(image_npt33),   0.0);
    	std::fill(std::begin(image_cpt47),   std::end(image_cpt47),   0.0);

        // Make image
        float deta, dphi;
        int w, h;
    	for(int d = 0; d < n_dau; ++d){
            deta = dau_deta->at(d);
            dphi = dau_dphi->at(d);

            if((abs(deta) > kDEtaMax) or (abs(dphi) > kDPhiMax))
                continue;

            std::tie(w, h) = CalcIndices(dau_deta->at(d), dau_dphi->at(d));

    	    // charged particle
	        if(dau_charge->at(d)){
	    	    image_cpt33[33*h + w] += float(dau_pt->at(d));
	    	    image_cpt47[47*h + w] += float(dau_pt->at(d));
        		image_cmult33[33*h + w] += 1.0;
	        }
            // neutral particle
	        else{
        		image_npt33[33*h + w] += float(dau_pt->at(d));
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

template<typename T>
T SumPixelIntensity(T image[], int size){
    T output = std::accumulate(
        image, // first
        image+size, //last: the range of elements to sum
        0, // init: initial value of the sum
        std::plus<T>()); // op: binary operation function
    return output;
}

template<typename T>
int CountNonZeroPixels(T image[], int size){
    int output = std::count_if(
        image,
        image+size,
        [](T p_){return p_ != 0;});
    return output;
}


void ScaleImage(float image[], int size, float scale_factor){
    std::transform(image,
                   image+size,
                   image,
                   [scale_factor](float v_)->float{return v_ / scale_factor;});
}


std::tuple<TString, TString, TString>
SplitNScale(TString const& input_path,
                     TString const& input_key="jetAnalyser"){

    std::cout << "\n#################################################" << endl;
    std::cout << "Input: " << input_path << endl;

    TFile* input_file = new TFile(input_path, "READ");
    TTree* input_tree = (TTree*) input_file->Get(input_key);
    const int input_entries = input_tree->GetEntries();

    #define SBA(name) input_tree->SetBranchAddress(#name, &name);
    float image_cpt33[33*33];  SBA(image_cpt33);
    float image_npt33[33*33];  SBA(image_npt33);
    float image_cmult33[33*33];  SBA(image_cmult33);
    float image_cpt47[47*47];  SBA(image_cpt47);


    std::map<TString, float> scale_factor;
    // 
    scale_factor["image_cpt33"] = 0.0;
    scale_factor["image_npt33"] = 0.0;
    scale_factor["image_cmult33"] = 0.0;
    scale_factor["image_cpt47"] = 0.0;

    std::map<TString, int> num_non_zero;
    num_non_zero["image_cpt33"] = 0;
    num_non_zero["image_npt33"] = 0;
    num_non_zero["image_cmult33"] = 0;
    num_non_zero["image_cpt47"] = 0;


    // val = validation
    const int val_start = int(input_entries*0.6);
    const int test_start = int(input_entries*0.8);

    // OUTPUTS
    TString input_dir = gSystem->DirName(input_path);
    TString parent_dir = gSystem->DirName(input_dir);

    TString output_dir = gSystem->ConcatFileName(parent_dir, "jet_image");
    if(gSystem->AccessPathName(output_dir))
        gSystem->mkdir(output_dir);

    TString input_name = gSystem->BaseName(input_path);
    TString name_fmt = input_name.Insert(input_name.Last('.'), "%s");

    TString train_name = TString::Format(name_fmt, "_training");
    TString val_name = TString::Format(name_fmt, "_validation");
    TString test_name = TString::Format(name_fmt, "_test");

    TString train_path = gSystem->ConcatFileName(output_dir, train_name);
    TString val_path = gSystem->ConcatFileName(output_dir, val_name);
    TString test_path = gSystem->ConcatFileName(output_dir, test_name);

    // training
    TFile* train_file = new TFile(train_path, "RECREATE");
    TTree* train_tree = input_tree->CloneTree(0);
    train_tree->SetDirectory(train_file);

    auto train_info = train_tree->GetUserInfo();

    // vaildiation
    TFile* val_file = new TFile(val_path, "RECREATE");
    TTree* val_tree = input_tree->CloneTree(0);
    val_tree->SetDirectory(val_file);

    // test
    TFile* test_file = new TFile(test_path, "RECREATE");
    TTree* test_tree = input_tree->CloneTree(0);
    test_tree->SetDirectory(test_file);

    // sum pixel intensity and count the # of non zero pixel.
    for(unsigned int i=0; i<val_start; i++){
        input_tree->GetEntry(i);

        scale_factor["image_cpt33"] += SumPixelIntensity(image_cpt33, 33*33);
        scale_factor["image_npt33"] += SumPixelIntensity(image_npt33, 33*33);
        scale_factor["image_cmult33"] += SumPixelIntensity(image_cmult33, 33*33);
        scale_factor["image_cpt47"] += SumPixelIntensity(image_cpt47, 47*47);

        num_non_zero["image_cpt33"] += CountNonZeroPixels(image_cpt33, 33*33);
        num_non_zero["image_npt33"] += CountNonZeroPixels(image_npt33, 33*33);
        num_non_zero["image_cmult33"] += CountNonZeroPixels(image_cmult33, 33*33);
        num_non_zero["image_cpt47"] += CountNonZeroPixels(image_cpt47, 47*47);
    }

    for(TString key : {"image_cpt33", "image_npt33", "image_cmult33", "image_cpt47"})
        scale_factor[key] /= num_non_zero[key];




    // Sclae training image and fill it. 
    for(unsigned int i=0; i<val_start; i++){
        input_tree->GetEntry(i);

        ScaleImage(image_cpt33,   33*33, scale_factor["image_cpt33"]);
        ScaleImage(image_npt33,   33*33, scale_factor["image_npt33"]);
        ScaleImage(image_cmult33, 33*33, scale_factor["image_cmult33"]);
        ScaleImage(image_cpt47,   47*47, scale_factor["image_cpt47"]);
        
        train_tree->Fill();
    }

    #define ADD_INFO(key) \
        TParameter<float>* param_##key = new TParameter<float>(TString(#key), scale_factor[#key]); \
        train_info->Add(param_##key);

    ADD_INFO(image_cpt33);
    ADD_INFO(image_npt33);
    ADD_INFO(image_cmult33);
    ADD_INFO(image_cpt47);

    for(unsigned int i=val_start; i<test_start; i++){
        input_tree->GetEntry(i);

        ScaleImage(image_cpt33,   33*33, scale_factor["image_cpt33"]);
        ScaleImage(image_npt33,   33*33, scale_factor["image_npt33"]);
        ScaleImage(image_cmult33, 33*33, scale_factor["image_cmult33"]);
        ScaleImage(image_cpt47,   47*47, scale_factor["image_cpt47"]);

        val_tree->Fill();
    }

    for(unsigned int i=test_start; i<input_entries; i++){
        input_tree->GetEntry(i);

        ScaleImage(image_cpt33,   33*33, scale_factor["image_cpt33"]);
        ScaleImage(image_npt33,   33*33, scale_factor["image_npt33"]);
        ScaleImage(image_cmult33, 33*33, scale_factor["image_cmult33"]);
        ScaleImage(image_cpt47,   47*47, scale_factor["image_cpt47"]);

        test_tree->Fill();
    }

    train_file->Write();
    val_file->Write();
    test_file->Write();


    std::cout << "Output: " << train_path << "," << endl;
    std::cout << "        " << val_path << "," << endl;
    std::cout << "        " << test_path << endl;
    std::cout << "\n#################################################" << endl;


    return std::make_tuple(train_path, val_path, test_path);
}



TString PrepTestData(TString const& input_path,
                     TString const& train_path){

    // File to be preprocessed
    std::cout << "\n#################################################" << endl;
    std::cout << "A File to be preprocessed: " << input_path << endl;
    std::cout << "A File having scale facotr: " << train_path << endl;

    TFile* input_file = new TFile(input_path, "READ");
    TTree* input_tree = (TTree*) input_file->Get("jetAnalyser");

    #define SBA(name) input_tree->SetBranchAddress(#name, &name);
    float image_cpt33[33*33];  SBA(image_cpt33);
    float image_npt33[33*33];  SBA(image_npt33);
    float image_cmult33[33*33];  SBA(image_cmult33);
    float image_cpt47[47*47];  SBA(image_cpt47);

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
    TString input_dir = gSystem->DirName(input_path);

    TString input_name = gSystem->BaseName(input_path);
    TString suffix = train_path.Contains("dijet") ? "_after_dijet" : "_after_zjet";
    TString output_name = input_name.Insert(input_name.Last('.'), suffix);
    TString output_path = gSystem->ConcatFileName(input_dir, output_name);

    TFile* output_file = new TFile(output_path, "RECREATE");
    TTree* output_tree = input_tree->CloneTree(0);
    output_tree->SetDirectory(output_file);

    const int input_entries = input_tree->GetEntries();
    for(unsigned int i=0; i<input_entries; i++){
        input_tree->GetEntry(i);

        ScaleImage(image_cpt33,   33*33, scale_factor["image_cpt33"]);
        ScaleImage(image_npt33,   33*33, scale_factor["image_npt33"]);
        ScaleImage(image_cmult33, 33*33, scale_factor["image_cmult33"]);
        ScaleImage(image_cpt47,   47*47, scale_factor["image_cpt47"]);

        output_tree->Fill();
    }

    output_file->Write();

    std::cout << "Output: " << output_path << "," << endl;
    std::cout << "\n#################################################" << endl;


    return output_path;
}


void macro(){
    TString data_dir = "../Data/FastSim_pt_100_500/jet_image";

    TString dijet_train = gSystem->ConcatFileName(data_dir, "dijet_training.root"); 
    TString dijet_validation = gSystem->ConcatFileName(data_dir, "dijet_validation.root"); 
    TString dijet_test = gSystem->ConcatFileName(data_dir, "dijet_test.root"); 

    TString zjet_train = gSystem->ConcatFileName(data_dir, "zjet_training.root"); 
    TString zjet_validation = gSystem->ConcatFileName(data_dir, "zjet_validation.root"); 
    TString zjet_test = gSystem->ConcatFileName(data_dir, "zjet_test.root"); 

    PrepTestData(dijet_validation, dijet_train);
    PrepTestData(dijet_test, dijet_train);
    PrepTestData(dijet_validation, zjet_train);
    PrepTestData(dijet_test, zjet_train);

    PrepTestData(zjet_validation, dijet_train);
    PrepTestData(zjet_test, dijet_train);
    PrepTestData(zjet_validation, zjet_train);
    PrepTestData(zjet_test, zjet_train);

}
