#include "TFile.h"
#include "TTree.h"
#include "TSystem.h"
#include "TString.h"

#include <iostream> // cout, endl
#include <vector> // std::vector
#include <map> // std::map
#include <numeric> // fill
#include <cmath> // pow, sqrt

#include "TimeUtils.cc"

const Float_t kDEtaMax = 0.4;
const Float_t kDPhiMax = 0.4;

// PDGId
const Int_t kElectronId = 11;
const Int_t kMuonId = 13;


inline Int_t pixelate(Float_t deta,
                      Float_t dphi,
                      Float_t deta_max=0.4,
                      Float_t dphi_max=0.4, 
                      Int_t num_bins=33)
{
    Int_t eta_idx = static_cast<Int_t>(num_bins * (deta + deta_max) / (2 * deta_max));
    Int_t phi_idx = static_cast<Int_t>(num_bins * (dphi + dphi_max) / (2 * dphi_max));
    Int_t idx = num_bins * eta_idx + phi_idx;
    return idx;
}


TString MakeJetImage(TString const& in_path,
                     TString const& out_dir,
                     TString const& tree_name="jetAnalyser")
{


    std::cout << "\n#################################################" << std::endl;
    std::cout << "Input: " << in_path << std::endl << std::endl;

    TFile* in_file = TFile::Open(in_path, "READ");
    TTree* in_tree = dynamic_cast<TTree*>( in_file->Get(tree_name) );
    const Int_t kInEntries = in_tree->GetEntries();
    std::cout << "  Input entries: " << kInEntries << std::endl;

    // daughters for jet image
    Int_t n_dau;
    std::vector<Float_t> *dau_pt=0, *dau_deta=0, *dau_dphi=0;
    std::vector<Int_t> *dau_charge=0, *dau_pid=0, *dau_ishadronic=0;

    #define SBA(name) in_tree->SetBranchAddress(#name, &name);
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
    if(gSystem->AccessPathName(out_dir))
        gSystem->mkdir(out_dir);

    TString out_name = gSystem->BaseName(in_path);
    TString out_path = gSystem->ConcatFileName(out_dir, out_name);

    TFile* out_file = TFile::Open(out_path, "RECREATE");
    TTree* out_tree = in_tree->CloneTree(0);
    out_tree->SetDirectory(out_file);

    // Image Branch
    std::vector<TString> image_names = {
        "image_cpt_33",
        "image_npt_33",
        "image_cmult_33",
        "image_nmult_33",
        "image_lepton_pt_33",
        "image_electron_pt_33",
        "image_muon_pt_33",
        "image_photon_pt_33",
        "image_chad_pt_33",
        "image_nhad_pt_33",
        "image_lepton_mult_33",
        "image_electron_mult_33",
        "image_muon_mult_33",
        "image_photon_mult_33",
        "image_chad_mult_33",
        "image_nhad_mult_33"
    };

    // Image size map
    std::map<TString, Int_t> image_size_map;
    for(auto each : image_names)
    {
        auto substr = each(each.Last('_') + 1, each.Sizeof());
        Int_t height = std::atoi(substr.Data());
        Int_t size = static_cast<Int_t>(std::pow(height, 2));
        image_size_map[each] = size;
    }


    std::map<TString, Float_t*> image_map;
    TString leaflist_fmt = "%s[%d]/F";
    for(auto each : image_names)
    {
        std::cout << each << std::endl;
        const Int_t kImageSize = image_size_map[each];
        Float_t* tmp_array = new Float_t[kImageSize];
        std::fill(tmp_array, tmp_array + kImageSize, 0.0);
        image_map[each] = tmp_array;
        // Branch
        out_tree->Branch(
            each, // branchname
            image_map[each], // address
            TString::Format(leaflist_fmt, each.Data(), image_size_map[each])
        ); // leaflist
    }

    /////////////////////////////
    // Make Jet Image
    ///////////////////////////////////
    Float_t deta, dphi, daughter_pt;
    Int_t daughter_charge, daughter_pid, daughter_ishadronic;

    Int_t idx;

    const Int_t kPrintFreq = static_cast<Int_t>( kInEntries / 10 );
    TString print_fmt = "[%d %] %d entry | Elapsed time: %lf sec";

    Timer timer;
    timer.Start();
    std::cout << "Make jet images." << std::endl;
    for(Int_t i = 0; i < kInEntries; ++i)
    {
    	in_tree->GetEntry(i);

        if( i % kPrintFreq == 0 )
            std::cout << TString::Format(print_fmt, i / kPrintFreq * 10, i, timer.GetElapsedTime()) << std::endl;

        for(auto it = image_map.begin(); it != image_map.end(); it++)
            std::fill(it->second, it->second + image_size_map[it->first], 0.0);

        // TODO FILL ZERO
    	for(Int_t d = 0; d < n_dau; d++)
        {
            deta = dau_deta->at(d);
            dphi = dau_dphi->at(d);

            if( std::fabs(deta) >= kDEtaMax ) continue;
            if( std::fabs(dphi) >= kDPhiMax ) continue;

            idx = pixelate(deta, dphi); 

            // pT of d-th constituent of a jet
            daughter_pt = dau_pt->at(d);
            daughter_charge = dau_charge->at(d);
            daughter_pid = dau_pid->at(d);
            daughter_ishadronic = dau_ishadronic->at(d);
    	    // charged particle
	        if(daughter_charge)
            { 
	    	    image_map["image_cpt_33"][idx] += daughter_pt;
        		image_map["image_cmult_33"][idx] += 1.0;
                // Electron or Positron
                if( std::abs( daughter_pid ) == kElectronId )
                {
                    image_map["image_electron_pt_33"][idx] += daughter_pt;
                    image_map["image_electron_mult_33"][idx] += 1.0;

                    image_map["image_lepton_pt_33"][idx] += daughter_pt;
                    image_map["image_lepton_mult_33"][idx] += 1.0;
                }
                // Muon or antimuon
                else if( std::abs( daughter_pid ) == kMuonId )
                {
                    image_map["image_muon_pt_33"][idx] += daughter_pt;
                    image_map["image_muon_mult_33"][idx] += 1.0;

                    image_map["image_lepton_pt_33"][idx] += daughter_pt;
                    image_map["image_lepton_mult_33"][idx] += 1.0;
                }
                // Charged Hadrons
                else
                {
                    image_map["image_chad_pt_33"][idx] += daughter_pt;
                    image_map["image_chad_mult_33"][idx] += 1.0;
                }
	        }
            // Neutral particle
	        else
            {
        		image_map["image_npt_33"][idx] += daughter_pt;
        		image_map["image_nmult_33"][idx] += 1.0;
                // Neutral Hadron
                if( daughter_ishadronic )
                {
                    image_map["image_nhad_pt_33"][idx] += daughter_pt;
                    image_map["image_nhad_mult_33"][idx] += 1.0;
                }
                // Photon
                else
                {
                    image_map["image_photon_pt_33"][idx] += daughter_pt;
                    image_map["image_photon_mult_33"][idx] += 1.0;
                }
    	    }

    	}
        out_tree->Fill();
    }
    std::cout << "Elapsed time is " << timer.GetElapsedTime() << std::endl;
    
    out_file->Write();
    out_file->Close();
    in_file->Close();

    std::cout << "Output: " << out_path << std::endl;
    return out_path; 
}
