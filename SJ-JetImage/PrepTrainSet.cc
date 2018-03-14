#include "TFile.h"
#include "TTree.h"
#include "TSystem.h"
#include "TString.h"
#include "TVectorF.h"

#include <iostream>
#include <vector> // std::vector
#include <map> // std::map
#include <algorithm> // std::transform 
#include <numeric> // std::accumulate
#include <cmath> // pow, sqrt

#include "TimeUtils.cc"

TString PrepTrainSet(TString const& in_path,
                     TString const& out_dir,
                     bool use_zero_centering,
                     bool use_standardization,
                     float const& r=1e-5)
{
    Timer global_timer(true);
    Timer local_timer(false);

    //////////////////////
    // Input
    //////////////////////
    std::cout << std::endl << GetNow();
    std::cout << std::endl << "##################################" << std::endl;
    std::cout << "Input: " << in_path << std::endl << std::endl;

    TFile* in_file = TFile::Open(in_path, "READ");
    TTree* in_tree = dynamic_cast<TTree*>(in_file->Get("jetAnalyser"));
    const Int_t kInEntries = in_tree->GetEntries();
    const Int_t kPrintFreq = static_cast<Int_t>( kInEntries / 10 );
    TString print_fmt = "[%d %] %d entry | Elapsed time: %lf sec";

    // Get the vector of image branch names
    std::vector<TString> image_names;
    TString branch_name;
    for(auto each : *(in_tree->GetListOfBranches()))
    {
        branch_name = each->GetName();
        if(branch_name.Contains("image"))
            image_names.push_back(branch_name);
    }

    // Image size map
    std::map<TString, Int_t> image_size_map;
    for(auto each : image_names)
    {
        auto substr = each(each.Last('_') + 1, each.Sizeof());
        Int_t height = std::atoi(substr.Data());
        Int_t size = static_cast<Int_t>(std::pow(height, 2));
        image_size_map[each] = size;
    }

    // SetBranchAddress
    std::map<TString, Float_t*> image_map;
    for(auto each : image_names)
    {
        // Create
        const int image_size = image_size_map[each];
        Float_t* tmp_array = new Float_t[image_size];
        std::fill(tmp_array, tmp_array + image_size, 0.0);
        image_map[each] = tmp_array;
        // Branch
        in_tree->SetBranchAddress(each, image_map[each]);
    }


    /////////////////////////
    // Output file and tree
    /////////////////////////////
    if(gSystem->AccessPathName(out_dir))
        gSystem->mkdir(out_dir);

    TString out_name = gSystem->BaseName(in_path);
    TString out_path = gSystem->ConcatFileName(out_dir, out_name);

    TFile* out_file = TFile::Open(out_path, "RECREATE");
    TTree* out_tree = in_tree->CloneTree(0);  
    out_tree->SetDirectory(out_file);


    //////////////////////////////////
    // Preprocessing
    //////////////////////////////////

    /******************************************
     *  Normalization
     ******************************************/
    std::cout << std::endl << std::endl << GetNow();
    std::cout << "[Normalization] Start the normalization." << std::endl;

    Float_t intensity_sum;
    local_timer.Start();
    for(Int_t i = 0; i < kInEntries; i++)
    {
        in_tree->GetEntry(i);
        if( i % kPrintFreq == 0 ) 
            std::cout << TString::Format(print_fmt, i / kPrintFreq * 10, i, local_timer.GetElapsedTime()) << std::endl;

        for(auto it = image_map.begin(); it != image_map.end(); it++)
        {
            intensity_sum = std::accumulate(it->second, it->second + image_size_map[it->first], 0);

            std::transform(it->second, // first1
                           it->second + image_size_map[it->first], // last1
                           it->second, // d_first
                           [intensity_sum](Float_t v_)->Float_t{return v_ / intensity_sum;}); // unary_op
        }
    }

    std::cout << "[Normalization] Finish off the normalization" << std::endl;
    std::cout << "                Elapsed time is " << local_timer.GetElapsedTime() << std::endl;

    //////////////////////////////
    // Zero-center
    /////////////////////////////////
    if(use_zero_centering)
    {
        Timer zc_timer(true);
        std::cout << std::endl << std::endl << GetNow();
        std::cout << "[Zero-Centering] Start the zero-centering" << std::endl;

        // Create the map of image mean
        std::map<TString, Float_t*> mean_map;
        TString mean_name;
        for(auto each : image_names)
        {
            std::cout << each << std::endl;
            const int image_size = image_size_map[each];
            Float_t* tmp_array = new Float_t[image_size];
            std::fill(tmp_array, tmp_array + image_size, 0.0);
            mean_map[each] = tmp_array; 
        }

        // Compute the mean of whole images
        std::cout << "[Zero-Centering] Sum I_{ij}" << std::endl; 
        local_timer.Reset();
        for(Int_t i = 0; i < kInEntries; i++)
        {
            in_tree->GetEntry(i);

            if( i % kPrintFreq == 0 )
                std::cout << TString::Format(print_fmt, i / kPrintFreq * 10, i, local_timer.GetElapsedTime()) << std::endl;

            for(auto each : image_names)
            {
                for(Int_t j=0; j < image_size_map[each]; j++)
                    mean_map[each][j] += image_map[each][j];
            }
        }
        std::cout << "[Zero-Centering] Finish off the summation." << std::endl << std::endl;
        std::cout << "                 Elapsed time is " << local_timer.GetElapsedTime() << std::endl << std::endl;
 
        std::cout << "[Zero-Centering] Divide the intensity sum by # of entries" << std::endl;
        local_timer.Reset();
        for(auto it = mean_map.begin(); it != mean_map.end(); it++)
        {
            std::transform(it->second,
                           it->second + image_size_map[it->first],
                           it->second,
                           [kInEntries](Float_t v_)->Float_t{return v_ / kInEntries;});
        }
        std::cout << "[Zero-Centring] Finish off the division." << std::endl;
        std::cout << "                 Elapsed time is " << local_timer.GetElapsedTime() << std::endl << std::endl;

        // Subtract mu from each images
        std::cout << "[Zero-Centering] Subtract the mean of normalized training images from each image." << std::endl;
        local_timer.Reset();
        for(Int_t i = 0; i < kInEntries; i++)
        {
            in_tree->GetEntry(i);

            if( i % kPrintFreq == 0 )
                std::cout << TString::Format(print_fmt, i / kPrintFreq * 10, i, local_timer.GetElapsedTime()) << std::endl;

            for(auto each : image_names)
            {
                for(Int_t j=0; j < image_size_map[each]; j++)
                    image_map[each][j] -= mean_map[each][j];
            }
        }
        std::cout << "[Zero-Centering] Finish off the subtraction." << std::endl;
        std::cout << "                 Elapsed time is " << local_timer.GetElapsedTime() << std::endl << std::endl;

        // Write the mean of images to out file.
        out_file->cd();
        out_file->mkdir("image_mean");
        out_file->cd("image_mean"); 
        TVectorF* tvec;
        for(auto it = mean_map.begin(); it != mean_map.end(); it++)
        {
            tvec = new TVectorF(image_size_map[it->first], it->second);
            tvec->Write(it->first);
        }
        out_file->cd();

        std::cout << "[Zero-Centering] Finish off the Zero-Centring" << std::endl;
        std::cout << "                 Elapsed time is " << zc_timer.GetElapsedTime() << std::endl;
    }

    /*****************************************
     * Standardization
     * Divide each pixel value by the standard deviation \sigma_{ij} of that
     * pixel value in the normalized training dataset,
     * I_{ij} \rightarrow I_{ij}/(\sigma_{ij}+r).
     * A value of r = 10^{-5} was usedto suppress noise.
     *********************************************/
    // sigma^2 = E[(I^2 - mu)^2] = E[I^2] because mu=0
    if( use_zero_centering and use_standardization )
    {
        Timer std_timer(true);
        std::cout << std::endl << std::endl << GetNow;
        std::cout << " [Standardization] Start the standardization." << std::endl;

        // Create stddev_map
        std::map<TString, Float_t*> stddev_map;
        TString stddev_name;
        for(auto each : image_names)
        {
            std::cout << each << std::endl;
            const int size = image_size_map[each];
            Float_t* tmp_array = new Float_t[size];
            std::fill(tmp_array, tmp_array + size, 0.0);
            stddev_map[each] = tmp_array; 
        }

        // Compute stddev_map
        std::cout << "[Standardization] Start to compute Sum[I^2]" << std::endl;
        local_timer.Reset();
        for(Int_t i = 0; i < kInEntries; i++)
        {
            if( i % kPrintFreq == 0 )
                std::cout << TString::Format(print_fmt, i / kPrintFreq * 10, i, local_timer.GetElapsedTime()) << std::endl;

            in_tree->GetEntry(i);
            for(auto each : image_names)
            {
                for(Int_t j = 0; j < image_size_map[each]; j++)
                    stddev_map[each][j] += std::pow(image_map[each][j], 2);
            }
        }
        std::cout << "[Standardization] Finish off Sum[I^2] computation" << std::endl;
        std::cout << "                  Elapsed time is " << local_timer.GetElapsedTime() << std::endl;


        std::cout << "[Standardization] Start to divide Sum[I^2] by N and . Result is StdE[I^]" << std::endl;
        local_timer.Reset();
        for(auto it = stddev_map.begin(); it != stddev_map.end(); it++)
        {
            std::transform(it->second,
                           it->second + image_size_map[it->first],
                           it->second,
                           [kInEntries](Float_t v_)->Float_t{return std::sqrt(v_ / kInEntries);});
        }
        std::cout << "[Standardization] Finish off." << std::endl;
        std::cout << "                  Elapsed time is " << local_timer.GetElapsedTime() << std::endl;

        // Divide each pixel value by stddev
        std::cout << "[Standardization] Start the computation: I_{ij} --> I_{ij} / (sigma_{ij} + r)" << std::endl;
        local_timer.Reset();
        for(Int_t i = 0; i < kInEntries; i++)
        {
            if( i % kPrintFreq == 0 )
                std::cout << TString::Format(print_fmt, i / kPrintFreq * 10, i, local_timer.GetElapsedTime()) << std::endl;

            in_tree->GetEntry(i);
            for(auto each : image_names)
            {
                for(Int_t j=0; j < image_size_map[each]; j++)
                    image_map[each][j] /= stddev_map[each][j] + r;
            }
        }
        std::cout << "[Standardization] Finish off the computation: I_{ij} --> I_{ij} / (sigma_{ij} + r)" << std::endl;
        std::cout << "                  Elapsed time is " << local_timer.GetElapsedTime() << std::endl;


        // Write the stddev of images to out file.
        out_file->mkdir("image_stddev");
        out_file->cd("image_stddev");
        TVectorF* tvec;
        for(auto it = stddev_map.begin(); it != stddev_map.end(); it++)
        {
            tvec = new TVectorF(image_size_map[it->first], it->second);
            std::cout << "[" << it->first << "] max: " << tvec->Max() << std::endl;
            tvec->Write(it->first);
        }
        out_file->cd();

        std::cout << "[Standardization] Finish off the standardization." << std::endl; 
        std::cout << "                  Elapsed time is " << std_timer.GetElapsedTime() << std::endl; 
    }



    // Fill
    std::cout << std::endl << std::endl << GetNow();
    std::cout << "Fiil the output tree." << std::endl;
    local_timer.Reset();
    for(int i = 0; i < kInEntries; i++)
    {
        if( i % kPrintFreq == 0 )
            std::cout << TString::Format(print_fmt, i / kPrintFreq * 10, i, local_timer.GetElapsedTime()) << std::endl;
        in_tree->GetEntry(i);
        out_tree->Fill();
    }
    std::cout << "Elapsed time is " << local_timer.GetElapsedTime() << std::endl; 

    out_file->Write();
    out_file->Print();
    out_file->Close();

    in_file->Close();

    global_timer.Print();

    return out_path;
}
