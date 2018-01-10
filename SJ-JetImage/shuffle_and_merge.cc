#include "utils.cc"

// Step1
TString AttachLabel(TString input_path,
                    TString output_dir){
    // Input
    TFile* input_file = new TFile(input_path, "READ");
    TTree* input_tree = (TTree*) input_file->Get("jetAnalyser");

    TString input_name = gSystem->BaseName(input_path);
    int input_entries = input_tree->GetEntries();

    bool kIsQQ = input_name.Contains("qq");
    bool kIsGG = input_name.Contains("gg");
    bool kIsZQ = input_name.Contains("zq");

    bool kIsQuarkJets = kIsQQ or kIsZQ;
    bool kIsDijet = kIsQQ or kIsGG;

    // Output
    TString output_name = input_name;
    TString output_path = gSystem->ConcatFileName(output_dir, output_name);

    TFile* output_file = new TFile(output_path, "RECREATE");
    TTree* output_tree = input_tree->CloneTree(0);
    output_tree->SetDirectory(output_file);
    
    int label[2];
    if(kIsQuarkJets){
        label[0] = 1;
        label[1] = 0;
    }
    else{
        label[0] = 0;
        label[1] = 1;
    }
    
    output_tree->Branch("label", &label, "label[2]/I");
    
    for(int i=0; i < input_entries; i++){
        input_tree->GetEntry(i);
        output_tree->Fill();
    }
    
    output_file->Write();
    output_file->Close();
    
    input_file->Close();
    
    return output_path;
}


TString Step1_AttachLabel(TString input_dir){

    TString parent_dir = gSystem->DirName(input_dir);
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
TString ShuffleTree(TString input_path,
                    TString output_dir,
                    TString input_numcycle="jetAnalyser"){

    TFile* input_file = new TFile(input_path, "READ");
    TTree* input_tree = (TTree*) input_file->Get(input_numcycle);
    Int_t input_entries = input_tree->GetEntries();

    TString output_filename = gSystem->BaseName(input_path);
    TString output_path = gSystem->ConcatFileName(output_dir, output_filename);

    TFile* output_file = new TFile(output_path, "RECREATE");
    TTree* output_tree = input_tree->CloneTree(0);
    output_tree->SetDirectory(output_file);

    int order[input_entries];
    for(unsigned int i=0; i<input_entries; i++)
        order[i] = i;

    std::random_shuffle(order, order+input_entries );


    const int print_freq = int(input_entries/20.0);

    for(unsigned int i=0; i<input_entries; i++){
        input_tree->GetEntry(order[i]);

        if((i%print_freq==0) or ((i+1)==input_entries)){
            cout << "(" << 20 * i / print_freq << "%) "
                 << i << "the entires" << endl;
        }

        output_tree->Fill();
    }

    output_file->Write();
    output_file->Close();

    input_file->Close();

    return output_path;
}


TString Step3_Shuffle(TString input_dir){

    TString parent_dir = gSystem->DirName(input_dir);
    TString output_dir = gSystem->ConcatFileName(parent_dir, "step3_shuffle");
    if(gSystem->AccessPathName(output_dir))
        gSystem->mkdir(output_dir);

    std::vector< TString > dijet_files = ListDir(input_dir, ".root", "dijet");
    std::vector< TString > zjet_files = ListDir(input_dir, ".root", "zjet");

    TString output_path;

    for(auto dj : dijet_files)
        output_path = ShuffleTree(dj, output_dir);
        cout << output_path << endl << endl;

    for(auto zj : zjet_files)
        output_path = ShuffleTree(zj, output_dir);
        cout << output_path << endl << endl;

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


void macro(TString input_path){


    TString step1_path = Step1_AttachLabel(input_path);

    TString step2_path = Step2_Merge(step1_path);

    TString step3_path = Step3_Shuffle(step2_path);

    TString step4_path = Step4_Merge(step3_path);

}
