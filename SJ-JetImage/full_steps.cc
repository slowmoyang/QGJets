#include "MakeJetImage.cc"
#include "SplitDataset.cc"
#include "PrepTrainSet.cc"
#include "PrepTestSet.cc"

#include "TString.h"
#include "TSystem.h"

#include <tuple>
#include <iostream>
#include <cstdlib> // abort

#include "getopt.h"


int main(int argc, char* argv[])
{
    // TODO Argument Parsing 
    TString in_dir, out_dir;
    bool use_zero_center=false, use_standardization=false;

    int c;
    while (1)
    {
        static struct option long_options[] =
        {
            {"in_dir", required_argument, nullptr, 'i'},
            {"out_dir", required_argument, nullptr, 'o'},
            {"use_zero_center", no_argument, nullptr, 'z'},
            {"use_standardization", no_argument, nullptr, 's'},
            {0, 0, 0, 0}
        };

        int option_index = 0;

        c = getopt_long(argc, argv, "i:o:zs", long_options, &option_index);

        if (c == -1)
            break;

        switch (c)
        {
            case 0:
                if (long_options[option_index].flag != 0)
                    break;
                std::cout << "Option: " << long_options[option_index].name;
                if (optarg)
                    std::cout << " with arg " << optarg;
                std::cout << std::endl;
                break;
            case 'i':
                std::cout << "option -i with value " << optarg << std::endl;
                in_dir = TString(optarg);
            case 'o':
                std::cout << "option -o with value " << optarg << std::endl;
                out_dir = TString(optarg);
            case 'z':
                std::cout << "option -z" << std::endl;
                use_zero_center = true;
                break;
            case 's':
                std::cout << "option -s" << std::endl;
                use_standardization = true;
                break;
            case '?':
                break;
            default:
                std::abort();
        }
    }

    if (optind < argc)
    {
        std::cout << "non-option ARGV-elements: ";
        while (optind < argc)
            std::cout << argv[optind++] << " ";
        std::cout << std::endl;
    }

    // TODO 
    // dj means Dijet and zj means Z+jet.

    std::cout << "##########################################################" << std::endl;
    std::cout << "# Step1: Make jet images                                 #" << std::endl;
    std::cout << "##########################################################" << std::endl;
    TString in_dj_path = gSystem->ConcatFileName(in_dir, "dijet.root");
    TString in_zj_path = gSystem->ConcatFileName(in_dir, "zjet.root");

    if(gSystem->AccessPathName(out_dir))
        gSystem->mkdir(out_dir);

    // Step1: Make jet images.
    TString dj_image_path = MakeJetImage(in_dj_path, out_dir);
    TString zj_image_path = MakeJetImage(in_zj_path, out_dir);

    // Step2: Split the dataset to training, validation and test set.
    std::cout << "##################################################################" << std::endl;
    std::cout << "# Step2: Split dataset into training, validation and test sets.  #" << std::endl;
    std::cout << "##################################################################" << std::endl;

    TString dj_train_path, dj_val_path, dj_test_path;
    std::tie(dj_train_path, dj_val_path, dj_test_path) = SplitDataset(dj_image_path, out_dir);

    TString zj_train_path, zj_val_path, zj_test_path;
    std::tie(zj_train_path, zj_val_path, zj_test_path) = SplitDataset(zj_image_path, out_dir);

    // Step3: Preprocess training set.
    std::cout << "##########################################################" << std::endl;
    std::cout << "# Step3: Preprocess Dijet and Z+jet training sets.       #" << std::endl;
    std::cout << "##########################################################" << std::endl;

    TString dj_dir = gSystem->ConcatFileName(out_dir, "dijet_set");
    TString dj_train_prep_path = PrepTrainSet(dj_train_path, dj_dir, use_zero_center, use_standardization);

    TString zj_dir = gSystem->ConcatFileName(out_dir, "zjet_set");
    TString zj_train_prep_path = PrepTrainSet(zj_train_path, zj_dir, use_zero_center, use_standardization);

    // Step4: Preprocess validation and test set.
    std::cout << "##########################################################" << std::endl;
    std::cout << "# Step4: Preprocess validation/ and test sets.           #" << std::endl;
    std::cout << "##########################################################" << std::endl;

    // 
    PrepTestSet(dj_val_path, dj_dir, dj_train_prep_path);
    PrepTestSet(dj_test_path, dj_dir, dj_train_prep_path);
    PrepTestSet(zj_val_path, dj_dir, dj_train_prep_path);
    PrepTestSet(zj_test_path, dj_dir, dj_train_prep_path);

    // 
    PrepTestSet(dj_val_path, zj_dir, zj_train_prep_path);
    PrepTestSet(dj_test_path, zj_dir, zj_train_prep_path);
    PrepTestSet(zj_val_path, zj_dir, zj_train_prep_path);
    PrepTestSet(zj_test_path, zj_dir, zj_train_prep_path);


    return 0;
}
