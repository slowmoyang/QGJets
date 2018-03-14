#include "TFile.h"
#include "TTree.h"
#include "TSystem.h"
#include "TString.h"
#include "TRandom.h"

#include <iostream>
#include <tuple> // std::tuple, std::tie, std::make_tuple

#include "TimeUtils.cc"


std::tuple<TString, TString, TString>
SplitDataset(TString const& in_path,
             TString const& out_dir)
{
    std::cout << std::endl << "#################################################" << std::endl;
    std::cout << "In: " << in_path << std::endl;

    TFile* in_file = TFile::Open(in_path, "READ");
    TTree* in_tree = dynamic_cast<TTree*>( in_file->Get("jetAnalyser") );
    const Int_t kInEntries = in_tree->GetEntries();

    // OUTPUTS
    TString in_name = gSystem->BaseName(in_path);
    TString name_fmt = in_name.Insert(in_name.Last('.'), "%s");

    // e.g. dijet_training_set.root
    TString train_name = TString::Format(name_fmt, "_training_set");
    TString val_name = TString::Format(name_fmt, "_validation_set");
    TString test_name = TString::Format(name_fmt, "_test_set");

    TString train_path = gSystem->ConcatFileName(out_dir, train_name);
    TString val_path = gSystem->ConcatFileName(out_dir, val_name);
    TString test_path = gSystem->ConcatFileName(out_dir, test_name);

    // training
    TFile* train_file = TFile::Open(train_path, "RECREATE");
    TTree* train_tree = in_tree->CloneTree(0);
    train_tree->SetDirectory(train_file);

    // vaildiation
    TFile* val_file = TFile::Open(val_path, "RECREATE");
    TTree* val_tree = in_tree->CloneTree(0);
    val_tree->SetDirectory(val_file);

    // test
    TFile* test_file = TFile::Open(test_path, "RECREATE");
    TTree* test_tree = in_tree->CloneTree(0);
    test_tree->SetDirectory(test_file);


    const Int_t kPrintFreq = static_cast<Int_t>( kInEntries / 10 );
    Double_t split_prob;
    Timer timer(true);
    for(Int_t i=0; i < kInEntries; i++){
        in_tree->GetEntry(i);

        if( i % kPrintFreq == 0 ) {
            std::cout << "[" << i / kPrintFreq * 10 << " %]"
                      << i << "th entry" << std::endl; 
            timer.Print();
        }

        split_prob = gRandom->Uniform(0, 1);
        if ( split_prob < 0.6) {
            train_tree->Fill();
        }
        else if ( split_prob < 0.8 ) {
            val_tree->Fill();
        }
        else {
            test_tree->Fill();
        }
    }
    timer.Print();

    train_file->Write();
    val_file->Write();
    test_file->Write();

    Int_t num_train = train_tree->GetEntries();
    Int_t num_val = val_tree->GetEntries();
    Int_t num_test = test_tree->GetEntries();


    TString result_fmt = "# of examples in %s set: %d (%lf %%)";
    std::cout << TString::Format(result_fmt, "training", num_train, static_cast<Float_t>(num_train) / kInEntries) << std::endl;
    std::cout << TString::Format(result_fmt, "validation", num_val, static_cast<Float_t>(num_val) / kInEntries) << std::endl;
    std::cout << TString::Format(result_fmt, "test", num_test, static_cast<Float_t>(num_test) / kInEntries) << std::endl;


    train_file->Close();
    val_file->Close();
    test_file->Close();

    in_file->Close();

    std::cout << "Out: " << train_path << "," << std::endl;
    std::cout << "        " << val_path << "," << std::endl;
    std::cout << "        " << test_path << std::endl;
    std::cout << "\n#################################################" << std::endl;

    return std::make_tuple(train_path, val_path, test_path);
}


