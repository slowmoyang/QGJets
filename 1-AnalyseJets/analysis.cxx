#include "TFile.h"
#include "TTree.h"
#include "TClonesArray.h"
#include "classes/DelphesClasses.h"
#include "TMath.h"

#include <iostream>
#include <string>
#include <memory>
#include <vector>

using std::string;

Double_t minJetPT = 20.0;
Double_t maxJetEta = 2.4;
Double_t dRCut = 0.3;
bool doHadMerge = false;
bool doEcalCl = false;

bool debug=false;

//for Ripley K function
Double_t KF1(double a, double b, double c, int d, int e) {
  return (pow(TMath::Pi()*a, 2))/((TMath::Pi() - std::acos((pow(a, 2) - pow(b, 2) - pow(c, 2))/(2*b*c)))*d*e);
}
Double_t KF2(double a, int b, int c) {
  return (pow(TMath::Pi()*a, 2))/(TMath::Pi()*b*c);
}

Double_t DeltaPhi(Double_t phi1, Double_t phi2) {
  static const Double_t kPI = TMath::Pi();
  static const Double_t kTWOPI = 2*TMath::Pi();
  Double_t x = phi1 - phi2;
  if(TMath::IsNaN(x)){
    std::cerr << "DeltaPhi function called with NaN" << std::endl;
    return x;
  }
  while (x >= kPI) x -= kTWOPI;
  while (x < -kPI) x += kTWOPI;
  return x;
}

Double_t DeltaR(Double_t deta, Double_t dphi) {
  return TMath::Sqrt(deta*deta + dphi*dphi);
}

struct cluster {
  double pt;
  double eta;
  double phi;

  int ishadronic;
};

bool isBalanced(TClonesArray * gen_jets);
bool passZjets(TClonesArray * jets, TClonesArray * muons, TClonesArray * electrons, int &nGoodJets);

void fillDaughters(
    Jet *jet,
    float &leading_dau_pt,
    float& leading_dau_eta,
	std::vector<float> &dau_pt,
    std::vector<float> &dau_deta,
    std::vector<float> &dau_dphi,
	std::vector<int> &dau_charge,
    std::vector<int> &dau_ishadronic,
    std::vector<float> &dau_eemfrac,
    std::vector<float> &dau_ehadfrac,
    std::vector<int> &dau_pid,
    std::vector<int> &dau_istrack,

    int& nmult,
    int& cmult);



void fillDaughters_HADMERGE(Jet *jet, float &leading_dau_pt, float& leading_dau_eta,
			    std::vector<float> &dau_pt, std::vector<float> &dau_deta, std::vector<float> &dau_dphi,
			    std::vector<int> &dau_charge, std::vector<int> &dau_ishadronic, int& nmult);
void fillDaughter_EcalCluster(Jet *jet, float &leading_dau_pt, float& leading_dau_eta,
			      std::vector<float> &dau_pt, std::vector<float> &dau_deta, std::vector<float> &dau_dphi,
			      std::vector<int> &dau_charge, std::vector<int> &dau_ishadronic,
			      int& nmult, int& cmult, std::vector<float> &test);

int main(int argc, char *argv[])
{
  int c;
  while ((c = getopt(argc, argv, "eh")) != -1) {
    switch (c) {
    case 'e':
      doEcalCl = true;
      std::cout << "ECAL clustering turned on" << std::endl;
      break;
    case 'h':
      doHadMerge = true;
      std::cout << "Hadronic merging turned on" << std::endl;
      break;
    case '?':
      std::cout << "Bad Option: " << optopt << std::endl;
      exit(12);
    }
  }

  if ((argc - optind) != 2) {
    std::cout << "Requires input and output root file. Optional -e flag turns on ecal clustering, -h flag turns on hadronic merging" << std::endl;
    return 1;
  }

  auto in = string{argv[optind]};
  auto out = string{argv[optind+1]};

  std::cout << "Processing '" << in << "' into '" << out << "'" << std::endl;

  auto inf = TFile::Open(in.c_str());
  auto intr = (TTree*) inf->Get("Delphes");
  intr->SetBranchStatus("*", true);

  if (doEcalCl)
    std::cout << "ECAL Clustering" << std::endl;
  if (doHadMerge)
    std::cout << "E/H Mergeing" << std::endl;
  
  TClonesArray *jets = 0;
  intr->SetBranchAddress("Jet", &jets);
  TClonesArray *gen_jets = 0;
  intr->SetBranchAddress("GenJet", &gen_jets);
  TClonesArray *particles = 0;
  intr->SetBranchAddress("Particle", &particles);

  TClonesArray *electrons = 0;
  intr->SetBranchAddress("Electron", &electrons);
  TClonesArray *muons = 0;
  intr->SetBranchAddress("Muon", &muons);

  TClonesArray *vertices = 0;
  intr->SetBranchAddress("Vertex", &vertices);
  
  auto outf = TFile::Open(out.c_str(), "RECREATE");
  auto outtr = new TTree{"jetAnalyser", "jetAnalyser"};
  // Matching Jason's jetAnalyser output
#define Branch_(type, name, suffix) type name = 0; outtr->Branch(#name, &name, #name "/" #suffix);
#define BranchI(name) Branch_(Int_t, name, I)
#define BranchF(name) Branch_(Float_t, name, F)
#define BranchO(name) Branch_(Bool_t, name, O)
#define BranchVF(name) std::vector<float> name; outtr->Branch(#name, "vector<float>", &name);
#define BranchVI(name) std::vector<int> name; outtr->Branch(#name, "vector<int>", &name);
  BranchI(nEvent);
  BranchI(nJets);
  BranchI(nGoodJets);
  BranchI(nPriVtxs);
  // Jet is order'th jet by pt ordering
  BranchI(order);

  BranchF(pt);
  BranchF(eta);
  BranchF(phi);
  
  BranchF(pt_dr_log);
  BranchF(ptD);
  BranchF(axis1);
  BranchF(axis2);
  BranchF(leading_dau_pt);
  BranchF(leading_dau_eta);
  BranchI(nmult);
  BranchI(cmult);
  BranchI(partonId);
  BranchI(flavorId);

  BranchI(flavorAlgoId);
  BranchI(flavorPhysId);

#ifdef WOOJIN_VARIABLES
  // BDT (woojin) variables
  BranchF(GeoMoment);
  BranchF(HalfPtMoment);
  BranchF(DRSquareMoment);
  BranchF(SmallDRPT);
  BranchF(MassMoment);
  BranchF(PTSquare);
  BranchF(ptDoubleCone);
  //BranchI(mult);
  //BranchF(rho); // energy density
  
  BranchF(innerring0to1);
  BranchF(innerring1to2);
  BranchF(innerring2to3);
  BranchF(innerring3to4);
  BranchF(outerring)
  BranchF(weighted_innerring0to1);
  BranchF(weighted_innerring1to2);
  BranchF(weighted_innerring2to3);
  BranchF(weighted_innerring3to4);
  BranchF(weighted_outerring);
  BranchF(drmax);
  // Track-based Energy-Energy-Correlation(EEC) angularity (beta is tunable)
  BranchF(ang_EEC_beta_0);
  BranchF(ang_EEC_beta_1);
  BranchF(ang_EEC_beta_2);
  BranchF(ang_EEC_beta_3);
  BranchF(ang_EEC_beta_4);
  BranchF(ang_EEC_beta_5);
  BranchF(ang_EEC_beta_6);
  BranchF(ang_EEC_beta_7);
  BranchF(ang_EEC_beta_8);
  BranchF(ang_EEC_beta_9);
  BranchF(ang_EEC_beta_10);
  // Geomoment of charged daughter inside jet shape and outside jet shape 
  BranchF(inner_charged_GeoMoment_0_1);
  BranchF(inner_charged_GeoMoment_1_2);
  BranchF(inner_charged_GeoMoment_2_3);
  BranchF(inner_charged_GeoMoment_3_4);
  BranchF(outer_charged_GeoMoment);
  // charged daughter sum
  BranchF(sum_track_pt);
  //Ripley's K function
  BranchF(RKF_01);
  BranchF(RKF_02);
  BranchF(RKF_03);
  BranchF(RKF_04);
  BranchF(RKF_05);
  BranchF(RKF_06);
  BranchF(RKF_07);
  
  BranchF(MRKF_CC_01);
  BranchF(MRKF_CC_02);
  BranchF(MRKF_CC_03);
  BranchF(MRKF_CC_04);
  BranchF(MRKF_CC_05);
  BranchF(MRKF_CC_06);
  BranchF(MRKF_CC_07);

  BranchF(MRKF_CN_01);
  BranchF(MRKF_CN_02);
  BranchF(MRKF_CN_03);
  BranchF(MRKF_CN_04);
  BranchF(MRKF_CN_05);
  BranchF(MRKF_CN_06);
  BranchF(MRKF_CN_07);

  BranchF(MRKF_NC_01);
  BranchF(MRKF_NC_02);
  BranchF(MRKF_NC_03);
  BranchF(MRKF_NC_04);
  BranchF(MRKF_NC_05);
  BranchF(MRKF_NC_06);
  BranchF(MRKF_NC_07);

  BranchF(MRKF_NN_01);
  BranchF(MRKF_NN_02);
  BranchF(MRKF_NN_03);
  BranchF(MRKF_NN_04);
  BranchF(MRKF_NN_05);
  BranchF(MRKF_NN_06);
  BranchF(MRKF_NN_07);

  BranchVF(RKF);
  BranchVF(MRKF_CC);
  BranchVF(MRKF_CN);
  BranchVF(MRKF_NC);
  BranchVF(MRKF_NN);

#endif

  BranchVF(dau_pt);
  BranchVF(dau_deta);
  BranchVF(dau_dphi);
  BranchVF(test);
  BranchVI(dau_charge);
  BranchVI(dau_ishadronic);
  /////////////////////
  BranchVF(dau_eemfrac);
  BranchVF(dau_ehadfrac);
  BranchVI(dau_pid);
  BranchVI(dau_istrack);
  ///////////////////////
  BranchI(n_dau);
  BranchO(matched);
  BranchO(balanced);

  BranchO(lepton_overlap);
  BranchO(pass_Zjets);

  bool firstTime = true, badHardGenSeen = false;
  for (size_t iev = 0; iev < intr->GetEntries(); ++iev) {
    intr->GetEntry(iev);
    nEvent = iev;
    nJets = jets->GetEntries();
    order = 0;
    if (vertices)
      nPriVtxs = vertices->GetEntries();
    else // no pileup mode
      nPriVtxs = 1;
    
    pass_Zjets = passZjets(jets, muons, electrons, nGoodJets);
    
    std::vector<const GenParticle*> hardGen;
    for (unsigned k = 0; k < particles->GetEntries(); ++k) {
      auto p = (const GenParticle *) particles->At(k);
      if (p->Status != 23) continue; // Status 23 is hard process parton in Pythia8
      //if (p->Status < 20 || p->Status > 29) continue; // All 20s are hard processes (not all of them make sense but this is how QGL does it)

      if (abs(p->PID) > 5 && p->PID != 21) continue; // consider quarks and gluons only
      hardGen.push_back(p);
      if (firstTime) {
	std::cout << "WARNING: ASSUMING PYTHIA8 HARDQCD GENERATION, ONLY 2 HARD PARTONS CONSIDERED" << std::endl;
	firstTime = false;
      }
      if (hardGen.size() == 2) break;
    }
    if (!badHardGenSeen && (hardGen.size() != 2)) {
      std::cout << "hardGen " << hardGen.size() << std::endl;
      badHardGenSeen = true;
    }

    balanced = isBalanced(jets);
    // std::cout << "EV ";
    //   std::cout << "hardGen " << hardGen[0]->PID << " " << hardGen[1]->PID << " " << " " << hardGen[0]->PT << " " << hardGen[1]->PT << " | " << hardGen[0]->Eta << " " << hardGen[1]->Eta << " | " << " " << hardGen[0]->Phi << " " << hardGen[1]->Phi << std::endl;
    for (unsigned j = 0; j < jets->GetEntries(); ++j) {
      lepton_overlap = false;
      if (j >= 2) balanced = false; // only top 2 balanced

      auto jet = (Jet*) jets->At(j);

      // some cuts, check pt
      if (jet->PT < minJetPT) continue;
      if (fabs(jet->Eta) > maxJetEta) continue;

      // match to hard process in pythia
      matched = false;
      const GenParticle *match = 0;
      float dRMin = 10.;
      for (auto& p : hardGen) {
      	float dR = DeltaR(jet->Eta - p->Eta, DeltaPhi(jet->Phi, p->Phi));
      	if (dR < dRMin && dR < dRCut) {
      	  matched = true;
      	  match = p;
      	}
      }
      // if (!matched) continue;
      // std::cout << "hardGen " << hardGen[0]->PID << " " << hardGen[1]->PID << " " << matched << " " << jet->PT << " " << hardGen[0]->PT << " " << hardGen[1]->PT << " | " << jet->Eta << " " << hardGen[0]->Eta << " " << hardGen[1]->Eta << " | " << " " << jet->Phi << " " << hardGen[0]->Phi << " " << hardGen[1]->Phi << std::endl;
      
      // check overlapping jets
      bool overlap = false;
      for (unsigned k = 0; k < jets->GetEntries(); ++k) {
	if (k == j) continue;
	auto otherJet = (Jet*) jets->At(k);
	float dR = DeltaR(jet->Eta - otherJet->Eta, DeltaPhi(jet->Phi, otherJet->Phi));
	if (dR < dRCut) overlap = true;
      }
      if (overlap) continue;

      // check overlap with lepton
      for (unsigned k = 0; k < electrons->GetEntries(); ++k) {
	auto ele = (Electron*) electrons->At(k);
	float dR = DeltaR(jet->Eta - ele->Eta, DeltaPhi(jet->Phi, ele->Phi));
	if (dR < dRCut) lepton_overlap = true;
      }
      for (unsigned k = 0; k < muons->GetEntries(); ++k) {
	auto mu = (Muon*) muons->At(k);
	float dR = DeltaR(jet->Eta - mu->Eta, DeltaPhi(jet->Phi, mu->Phi));
	if (dR < dRCut) lepton_overlap = true;
      }

      pt = jet->PT;
      eta = jet->Eta;
      phi = jet->Phi;
      nmult = jet->NNeutrals;
      cmult = jet->NCharged;
      if (match)
	partonId = match->PID;
      else
	partonId = 0;
      flavorId = jet->Flavor;
      flavorAlgoId = jet->FlavorAlgo;
      flavorPhysId = jet->FlavorPhys;

#ifdef WOOJIN_VARIABLES
      ptDoubleCone = 0;
      GeoMoment = 0;
      HalfPtMoment = 0;
      DRSquareMoment = 0;
      SmallDRPT = 0;
      MassMoment = 0;
      PTSquare = 0;
      innerring0to1 = 0;
      innerring1to2 = 0;
      innerring2to3 = 0;
      innerring3to4 = 0;
      outerring = 0;
      weighted_innerring0to1 = 0;
      weighted_innerring1to2 = 0;
      weighted_innerring2to3 = 0;
      weighted_innerring3to4 = 0;
      weighted_outerring = 0;
      
      ang_EEC_beta_0 = 0;
      ang_EEC_beta_1 = 0;
      ang_EEC_beta_2 = 0;
      ang_EEC_beta_3 = 0;
      ang_EEC_beta_4 = 0;
      ang_EEC_beta_5 = 0;
      ang_EEC_beta_6 = 0;
      ang_EEC_beta_7 = 0;
      ang_EEC_beta_8 = 0;
      ang_EEC_beta_9 = 0;
      ang_EEC_beta_10 = 0;
      
      inner_charged_GeoMoment_0_1 = 0;
      inner_charged_GeoMoment_1_2 = 0;
      inner_charged_GeoMoment_2_3 = 0;
      inner_charged_GeoMoment_3_4 = 0;
      outer_charged_GeoMoment = 0;
      
      sum_track_pt = 0;
      drmax = 0;
      
      RKF_01 = 0;
      RKF_02 = 0;
      RKF_03 = 0;
      RKF_04 = 0;
      RKF_05 = 0;
      RKF_06 = 0;
      RKF_07 = 0;

      MRKF_CC_01 = 0;
      MRKF_CC_02 = 0;
      MRKF_CC_03 = 0;
      MRKF_CC_04 = 0;
      MRKF_CC_05 = 0;
      MRKF_CC_06 = 0;
      MRKF_CC_07 = 0;

      MRKF_CN_01 = 0;
      MRKF_CN_02 = 0;
      MRKF_CN_03 = 0;
      MRKF_CN_04 = 0;
      MRKF_CN_05 = 0;
      MRKF_CN_06 = 0;
      MRKF_CN_07 = 0;

      MRKF_NC_01 = 0;
      MRKF_NC_02 = 0;
      MRKF_NC_03 = 0;
      MRKF_NC_04 = 0;
      MRKF_NC_05 = 0;
      MRKF_NC_06 = 0;
      MRKF_NC_07 = 0;

      MRKF_NN_01 = 0;
      MRKF_NN_02 = 0;
      MRKF_NN_03 = 0;
      MRKF_NN_04 = 0;
      MRKF_NN_05 = 0;
      MRKF_NN_06 = 0;
      MRKF_NN_07 = 0;

      RKF.clear();
      MRKF_CC.clear();
      MRKF_CN.clear();
      MRKF_NC.clear();
      MRKF_NN.clear();

#endif


      axis1 = 0; axis2 = 0;
      ptD = 0;
      pt_dr_log = 0;
      dau_pt.clear();
      dau_deta.clear();
      dau_dphi.clear();
      dau_charge.clear();
      dau_ishadronic.clear();
      dau_eemfrac.clear();
      dau_ehadfrac.clear();
      dau_pid.clear();
      dau_istrack.clear();
      test.clear();

      if (doEcalCl)
	      fillDaughter_EcalCluster(jet, leading_dau_pt, leading_dau_eta, dau_pt, dau_deta, dau_dphi, dau_charge, dau_ishadronic, nmult, cmult, test);
      else if (doHadMerge)
	      fillDaughters_HADMERGE(jet, leading_dau_pt, leading_dau_eta, dau_pt, dau_deta, dau_dphi, dau_charge, dau_ishadronic, nmult);
      else
	      fillDaughters(jet,
                        leading_dau_pt,
                        leading_dau_eta,
                        dau_pt,
                        dau_deta,
                        dau_dphi,
                        dau_charge,
                        dau_ishadronic,
                        dau_eemfrac,
                        dau_ehadfrac,
                        dau_pid,
                        dau_istrack,
                        nmult,
                        cmult);

#ifdef WOOJIN_VARIABLES
        // charged dau pt sum
      for (size_t a = 0; a < dau_pt.size(); ++ a) {
        auto dau_a = jet->Constituents.At(a);
	auto track_a = dynamic_cast<Track*>(dau_a);
	if (track_a) {
	  sum_track_pt += track_a->PT;
	}
      }
      
        // radial distance of daugther farthest from the jet
      for (size_t b = 0; b< dau_pt.size(); ++b){
        double deta_maxcand = dau_deta[b];
        double dphi_maxcand = dau_dphi[b];
        double maxcand = DeltaR(deta_maxcand, dphi_maxcand);
        if (maxcand > drmax) {
          drmax = maxcand;
        }
      }
        // Ripley's K function estimater
      for ( int t = 0; t < 81; ++t){
        double sum_RKF = 0.;
        double sum_MRKF_CC = 0.;
        double sum_MRKF_CN = 0.;
        double sum_MRKF_NC = 0.;
        double sum_MRKF_NN = 0.;
        double search_distance = ((double)t/100. + 0.01);
        // unvariate case and bivariate(charged, neutral) case
        for (size_t c = 0; c < dau_pt.size(); ++c ) {
          for (size_t d = 0; d < dau_pt.size(); ++d ) {
            auto dau_c = jet->Constituents.At(c);
            auto dau_d = jet->Constituents.At(d);
            double deta_c = dau_deta[c];
            double dphi_c = dau_dphi[c];
            double deta_d = dau_deta[d];
            double dphi_d = dau_dphi[d];
            double dr_c = DeltaR(deta_c, dphi_c);
            double dr_d = DeltaR(deta_d, dphi_d);
            double study_area = 0.8;
            auto cdau_c = dynamic_cast<Track*>(dau_c);
            auto ndau_c = dynamic_cast<Tower*>(dau_c);
            auto cdau_d = dynamic_cast<Track*>(dau_d);
            auto ndau_d = dynamic_cast<Tower*>(dau_d);
            if (c == d) continue;
            if (study_area < dr_c) continue;
            if (study_area < dr_d) continue;
            if (cdau_c && cdau_d){
              double dr_cc_cd = DeltaR(cdau_c->Eta - cdau_d->Eta, cdau_c->Phi - cdau_d->Phi);
              if (dr_cc_cd < search_distance){
                if (dr_cc_cd > study_area - dr_c){
                  sum_RKF += KF1(study_area, dr_c, dr_cc_cd, cmult + nmult, cmult + nmult);
                  sum_MRKF_CC += KF1(study_area, dr_c, dr_cc_cd, cmult, cmult);
                }
                if (dr_cc_cd <= study_area - dr_c){
                  sum_RKF += KF2(study_area, cmult + nmult, cmult + nmult);
                  sum_MRKF_CC += KF2(study_area, cmult, cmult);
                }
              }
            }
            if (cdau_c && ndau_d){
              double dr_cn_cd = DeltaR(cdau_c->Eta - ndau_d->Eta, cdau_c->Phi - ndau_d->Phi);
              if (dr_cn_cd < search_distance){
                if (dr_cn_cd > study_area - dr_c){
                  sum_RKF += KF1(study_area, dr_c, dr_cn_cd, cmult + nmult, cmult + nmult);
                  sum_MRKF_CN += KF1(study_area, dr_c, dr_cn_cd, cmult, nmult);
                }
                if (dr_cn_cd <= study_area - dr_c){
                  sum_RKF += KF2(study_area, cmult + nmult, cmult + nmult);
                  sum_MRKF_CN += KF2(study_area, cmult,nmult);
                }
              }
            }
            if (ndau_c && cdau_d){
              double dr_nc_cd = DeltaR(ndau_c->Eta - cdau_d->Eta, ndau_c->Phi - cdau_d->Phi);
              if (dr_nc_cd < search_distance){
                if (dr_nc_cd > study_area - dr_c){
                  sum_RKF += KF1(study_area, dr_c, dr_nc_cd, cmult + nmult, cmult + nmult);
                  sum_MRKF_NC += KF1(study_area, dr_c, dr_nc_cd, cmult, nmult);
                }
                if (dr_nc_cd <= study_area - dr_c){
                  sum_RKF += KF2(study_area, cmult + nmult, cmult + nmult);
                  sum_MRKF_NC += KF2(study_area, cmult, nmult);
                }
              }
            }
            if (ndau_c && ndau_d){
              double dr_nn_cd = DeltaR(ndau_c->Eta - ndau_d->Eta, ndau_c->Phi - ndau_d->Phi);
              if (dr_nn_cd < search_distance){
                if (dr_nn_cd > study_area - dr_c){
                  sum_RKF += KF1(study_area, dr_c, dr_nn_cd, cmult + nmult, cmult + nmult);
                  sum_MRKF_NN += KF1(study_area, dr_c, dr_nn_cd, nmult, nmult);
                }
                if (dr_nn_cd <= study_area - dr_c){
                  sum_RKF += KF2(study_area, cmult + nmult, cmult + nmult);
                  sum_MRKF_NN += KF2(study_area, nmult, nmult);
                }
              }
            }
          }
        }
        //std::cout << "search_distance: " << search_distance << ", RKF: " << sum_RKF << ", MRKF_CC: " << sum_MRKF_CC << ", MRKF_CN: " << sum_MRKF_CN << ", MRKF_NC: " << sum_MRKF_NC << ", MRKF_NN: " << sum_MRKF_NN << std::endl;
        if (search_distance == 0.1){
          RKF_01 = sum_RKF;
          MRKF_CC_01 = sum_MRKF_CC;
          MRKF_CN_01 = sum_MRKF_CN;
          MRKF_NC_01 = sum_MRKF_NC;
          MRKF_NN_01 = sum_MRKF_NN;
        }
        if (search_distance == 0.2){
          RKF_02 = sum_RKF;
          MRKF_CC_02 = sum_MRKF_CC;
          MRKF_CN_02 = sum_MRKF_CN;
          MRKF_NC_02 = sum_MRKF_NC;
          MRKF_NN_02 = sum_MRKF_NN;
        }
        if (search_distance == 0.31){
          RKF_03 = sum_RKF;
          MRKF_CC_03 = sum_MRKF_CC;
          MRKF_CN_03 = sum_MRKF_CN;
          MRKF_NC_03 = sum_MRKF_NC;
          MRKF_NN_03 = sum_MRKF_NN;
        }
        if (search_distance == 0.4){
          RKF_04 = sum_RKF;
          MRKF_CC_04 = sum_MRKF_CC;
          MRKF_CN_04 = sum_MRKF_CN;
          MRKF_NC_04 = sum_MRKF_NC;
          MRKF_NN_04 = sum_MRKF_NN;
        }
        if (search_distance == 0.5){
          RKF_05 = sum_RKF;
          MRKF_CC_05 = sum_MRKF_CC;
          MRKF_CN_05 = sum_MRKF_CN;
          MRKF_NC_05 = sum_MRKF_NC;
          MRKF_NN_05 = sum_MRKF_NN;
        }
        if (search_distance == 0.6){
          RKF_06 = sum_RKF;
          MRKF_CC_06 = sum_MRKF_CC;
          MRKF_CN_06 = sum_MRKF_CN;
          MRKF_NC_06 = sum_MRKF_NC;
          MRKF_NN_06 = sum_MRKF_NN;
        }
        if (search_distance == 0.7){
          RKF_07 = sum_RKF;
          MRKF_CC_07 = sum_MRKF_CC;
          MRKF_CN_07 = sum_MRKF_CN;
          MRKF_NC_07 = sum_MRKF_NC;
          MRKF_NN_07 = sum_MRKF_NN;
        }
        RKF.push_back(sum_RKF);
        MRKF_CC.push_back(sum_MRKF_CC);
        MRKF_CN.push_back(sum_MRKF_CN);
        MRKF_NC.push_back(sum_MRKF_NC);
        MRKF_NN.push_back(sum_MRKF_NN);
      }

#endif

      float sum_weight = 0;
      float sum_pt = 0;
      float sum_detadphi = 0;
      float sum_deta = 0, sum_deta2 = 0, ave_deta = 0, ave_deta2 = 0;
      float sum_dphi = 0, sum_dphi2 = 0, ave_dphi = 0, ave_dphi2 = 0;
      for (size_t ic = 0; ic < dau_pt.size(); ++ic) {
	double dpt = dau_pt[ic];
	double deta = dau_deta[ic];
	double dphi = dau_dphi[ic];
	double weight = dpt*dpt;	
 	double dr = DeltaR(deta, dphi);
	
	dr = DeltaR(deta, dphi);
	pt_dr_log += std::log(dpt / dr);
	sum_weight += weight;
	sum_pt += dpt;
	sum_deta += deta*weight;
	sum_deta2 += deta*deta*weight;
	sum_dphi += dphi*weight;
	sum_dphi2 += dphi*dphi*weight;
	sum_detadphi += deta*dphi*weight;

#ifdef WOOJIN_VARIABLES
        GeoMoment += (powf((dpt/pt),1))*(powf((dr/dRCut),1));
        HalfPtMoment += (powf((dpt/pt),1.5))*(powf((dr/dRCut),0));
        DRSquareMoment += (powf((dpt/pt),0))*(powf((dr/dRCut),2));
        if(dr < 0.1)
          SmallDRPT += (powf((dpt/pt),1))*(powf((dr/dRCut),0));
        MassMoment += (powf((dpt/pt),1))*(powf((dr/dRCut),2)); // At natural units (c == 1), energy, mass and momentum have the same dimension. So pt*(dR^2) can be considered like I = m*(r^2) ( : moment of inertia ). I think MassMoment means moment of inertia.
        PTSquare += (powf((dpt/pt),2))*(powf((dr/dRCut),0));
        if(dr < dRCut * 2)
          ptDoubleCone += dpt;

        if(dr < 0.1 && dr > 0.)
          innerring0to1 += (dpt*dr)/cmult;
        if(dr < 0.2 && dr > 0.1)
          innerring1to2 += (dpt*dr)/cmult;
        if(dr < 0.3 && dr > 0.2)
          innerring2to3 += (dpt*dr)/cmult;
        if(dr < 0.4 && dr > 0.3)
          innerring3to4 += (dpt*dr)/cmult;
        if(dr < dRCut*2 && dr > 0.4)
          outerring += (dpt*dr)/cmult;

        weighted_innerring0to1 = innerring0to1/drmax;
        weighted_innerring1to2 = innerring1to2/drmax;
        weighted_innerring2to3 = innerring2to3/drmax;
        weighted_innerring3to4 = innerring3to4/drmax;
        weighted_outerring = outerring/drmax;

        auto dau_ic = jet->Constituents.At(ic);
	auto track_ic = dynamic_cast<Track*>(dau_ic);
	if (track_ic) {
	  if (dr < 0.1 && dr > 0.) {
	    inner_charged_GeoMoment_0_1 += (powf((track_ic->PT/pt),1))*(powf((dr/dRCut),1));
	  }
	  if (dr < 0.2 && dr > 0.1) {
	    inner_charged_GeoMoment_1_2 += (powf((track_ic->PT/pt),1))*(powf((dr/dRCut),1));
	  }
	  if (dr < 0.3 && dr > 0.2) {
	    inner_charged_GeoMoment_2_3 += (powf((track_ic->PT/pt),1))*(powf((dr/dRCut),1));
	  }
	  if (dr < 0.4 && dr > 0.3) {
	    inner_charged_GeoMoment_3_4 += (powf((track_ic->PT/pt),1))*(powf((dr/dRCut),1));
	  }
	  if (dr < dRCut*2 && dr > 0.4) {
	    outer_charged_GeoMoment += (powf((track_ic->PT/pt),1))*(powf((dr/dRCut),1));
	  }
	  
	  for (size_t i = ic+1; i < dau_pt.size(); ++i) {
	    auto dau_i = jet->Constituents.At(i);
	    auto track_i = dynamic_cast<Track*>(dau_i);
	    if (track_i) {
	      ang_EEC_beta_0 += (track_ic->PT*track_i->PT)*(powf(DeltaR(track_ic->Eta - track_i->Eta, track_ic->Phi - track_i->Phi), 0))/(sum_track_pt);
	      ang_EEC_beta_1 += (track_ic->PT*track_i->PT)*(powf(DeltaR(track_ic->Eta - track_i->Eta, track_ic->Phi - track_i->Phi), 0.1))/(sum_track_pt);
	      ang_EEC_beta_2 += (track_ic->PT*track_i->PT)*(powf(DeltaR(track_ic->Eta - track_i->Eta, track_ic->Phi - track_i->Phi), 0.2))/(sum_track_pt);
	      ang_EEC_beta_3 += (track_ic->PT*track_i->PT)*(powf(DeltaR(track_ic->Eta - track_i->Eta, track_ic->Phi - track_i->Phi), 0.3))/(sum_track_pt);
	      ang_EEC_beta_4 += (track_ic->PT*track_i->PT)*(powf(DeltaR(track_ic->Eta - track_i->Eta, track_ic->Phi - track_i->Phi), 0.4))/(sum_track_pt);
	      ang_EEC_beta_5 += (track_ic->PT*track_i->PT)*(powf(DeltaR(track_ic->Eta - track_i->Eta, track_ic->Phi - track_i->Phi), 0.5))/(sum_track_pt);
	      ang_EEC_beta_6 += (track_ic->PT*track_i->PT)*(powf(DeltaR(track_ic->Eta - track_i->Eta, track_ic->Phi - track_i->Phi), 0.6))/(sum_track_pt);
	      ang_EEC_beta_7 += (track_ic->PT*track_i->PT)*(powf(DeltaR(track_ic->Eta - track_i->Eta, track_ic->Phi - track_i->Phi), 0.7))/(sum_track_pt);
	      ang_EEC_beta_8 += (track_ic->PT*track_i->PT)*(powf(DeltaR(track_ic->Eta - track_i->Eta, track_ic->Phi - track_i->Phi), 0.8))/(sum_track_pt);
	      ang_EEC_beta_9 += (track_ic->PT*track_i->PT)*(powf(DeltaR(track_ic->Eta - track_i->Eta, track_ic->Phi - track_i->Phi), 0.9))/(sum_track_pt);
	      ang_EEC_beta_10 += (track_ic->PT*track_i->PT)*(powf(DeltaR(track_ic->Eta - track_i->Eta, track_ic->Phi - track_i->Phi), 1.0))/(sum_track_pt);
	    }
	  }
	}
#endif

      }
      float a = 0, b = 0, c = 0;
      if (sum_weight > 0) {
	ptD = TMath::Sqrt(sum_weight) / sum_pt;

	ave_deta = sum_deta / sum_weight;
	ave_deta2 = sum_deta2 / sum_weight;
	ave_dphi = sum_dphi / sum_weight;
	ave_dphi2 = sum_dphi2 / sum_weight;

	a = ave_deta2 - ave_deta*ave_deta;
	b = ave_dphi2 - ave_dphi*ave_dphi;
	c = -(sum_detadphi/sum_weight - ave_deta*ave_dphi);

	float delta = sqrt(fabs((a-b)*(a-b) + 4*c*c));
	axis1 = (a+b+delta > 0) ? sqrt(0.5*(a+b+delta)) : 0;
	axis2 = (a+b-delta > 0) ? sqrt(0.5*(a+b-delta)) : 0;
      }

      // axis1 = -std::log(axis1);
      // axis2 = -std::log(axis2);

      n_dau = dau_pt.size();

      order++;
      outtr->Fill();
    }
  }

  outtr->Write();
  outf->Close();
  
  return 0;
}

/* Is the event balanced according to the criteria of pg 13 of http://cds.cern.ch/record/2256875/files/JME-16-003-pas.pdf */
bool isBalanced(TClonesArray * jets)
{
  if (jets->GetEntries() < 2) return false;

  auto obj1 = (Jet*) jets->At(0);
  auto obj2 = (Jet*) jets->At(1);

  // 2 jets of 30 GeV
  if (obj1->PT < 30.0) return false;
  if (obj2->PT < 30.0) return false;
  // that are back-to-back
  if (obj1->P4().DeltaPhi(obj2->P4()) < 2.5) return false;

  // and any 3rd jet requires pt < 30% of the avg. of the 2 leading jets
  if (jets->GetEntries() > 2) {
    auto obj3 = (Jet*) jets->At(2);
    return (obj3->PT < 0.3*(0.5*(obj2->PT  + obj1->PT)));
  } else return true;
}

/* Does the event pass the Zjets criteria according to the criteria of pg 11-12 of http://cds.cern.ch/record/2256875/files/JME-16-003-pas.pdf */
bool passZjets(TClonesArray * jets, TClonesArray * muons, TClonesArray * electrons, int &nGoodJets)
{
  bool pass_Zjets = false;

  if (jets->GetEntries() < 1) return false;


  nGoodJets = 0;
  int iMaxPt = -1;
  for (unsigned k = 0; k < jets->GetEntries(); ++k) {
    auto j = (const Jet *) jets->At(k);
    if (j->PT < minJetPT)
      continue;
    if (fabs(j->Eta) > maxJetEta)
      continue;
    if (iMaxPt < 0) iMaxPt = k;
    if (((const Jet*) jets->At(iMaxPt))->PT < j->PT)
      iMaxPt = k;
    
    nGoodJets++;
  }
  if (iMaxPt < 0) return false;

  // check for Z event
  TLorentzVector theDimuon;
  for (unsigned k = 0; k < muons->GetEntries(); ++k) {
    if (iMaxPt < 0) break;
    auto mu = (Muon*) muons->At(k);
    if (mu->PT < 20.) continue;
    for (unsigned kk = k; kk < muons->GetEntries(); ++kk) {
      auto mu2 = (Muon*) muons->At(kk);
      if (mu2->PT < 20.) continue;
      if (mu->Charge*mu2->Charge > 0) continue;
      auto dimuon = (mu->P4() + mu2->P4());
      if (dimuon.M() < 70. || dimuon.M() > 110.) continue;
      pass_Zjets = true;
      theDimuon = dimuon;
    }
  }

  // The paper doesn't consider the electron channel
  
  // for (unsigned k = 0; k < electrons->GetEntries(); ++k) {
  //   if (iMaxPt < 0) break;
  //   auto mu = (Electron*) electrons->At(k);      
  //   for (unsigned kk = k; kk < electrons->GetEntries(); ++kk) {
  //     auto mu2 = (Electron*) electrons->At(kk);
  //     if (mu->Charge*mu2->Charge > 0) continue;
  //     auto dimuon = (mu->P4() + mu2->P4());
  //     if (dimuon.M() < 60 || dimuon.M() > 120) continue;
  //     pass_Zjets = true;
  //     theDimuon = dimuon;
  //   }
  // }

  if (pass_Zjets) {
    auto j = (const Jet *) jets->At(iMaxPt);
    // require them to be back to back
    if (DeltaPhi(j->Phi, theDimuon.Phi()) < 2.5)
      pass_Zjets = false;
    for (unsigned k = 0; k < jets->GetEntries(); ++k) {
      if (k == iMaxPt) continue;
      auto j = (const Jet *) jets->At(k);
      if (j->PT > 0.3*theDimuon.Pt())
	pass_Zjets = false;
    }
  }

  return pass_Zjets;
}


void fillDaughter_EcalCluster(Jet *jet, float& leading_dau_pt, float& leading_dau_eta,
			      std::vector<float> &dau_pt, std::vector<float> &dau_deta, std::vector<float> &dau_dphi,
			      std::vector<int> &dau_charge, std::vector<int> &dau_ishadronic,
			      int& nmult, int& cmult, std::vector<float> &test)
{
  nmult = 0; cmult = 0;
  size_t n_dau = jet->Constituents.GetEntries();
  // Find ECal clusters
  std::vector<int> ecalDau;
  std::vector<int> hcalDau;

  leading_dau_pt = leading_dau_eta = 0;
  
  for (size_t ic = 0; ic < n_dau; ++ic) {
    double dpt, deta, dphi;
    auto dau = jet->Constituents.At(ic);
    // Constituents can be a tower (neutral) or a track (charged)
    auto tower = dynamic_cast<Tower*>(dau);
    if (!tower) {
      auto track = dynamic_cast<Track*>(dau);
      if (!track) {
	std::cout << "ERROR: Daughter not tower or track" << std::endl;
	exit(2);
      }

      // veto tracks from pileup, based on z-spread (see pileup card for Z0 spread)
      // if (fabs(track->Z) > 1.5*0.25) {
      // 	continue;
      // }
      test.push_back(track->Z);
      
      cmult++;
      dau_pt.push_back(track->PT);
      dau_deta.push_back(track->Eta - jet->Eta);
      dau_dphi.push_back(DeltaPhi(track->Phi, jet->Phi));
      dau_charge.push_back(track->Charge);
      dau_ishadronic.push_back(0);

      if (track->PT > leading_dau_pt) {
	leading_dau_pt = track->PT;
	leading_dau_eta = track->Eta - jet->Eta;
      }
      
      continue;
    }
    if (tower->ET < 1.0) { // Don't accept low energy neutrals
      continue;
    }
    
    if ((tower->Ehad != 0.0) && (tower->Eem != 0.0))
      std::cout << "ERROR: Tower with Had " << tower->Ehad << " and EM " << tower->Eem << " energy" << std::endl;

    if (tower->Ehad == 0.0) {
      ecalDau.push_back(ic);
    } else {
      hcalDau.push_back(ic);
    }
  }

  // sort by pt of the cell
  std::sort(ecalDau.begin(), ecalDau.end(),
	    [&jet](int a, int b){ return ((Tower*) jet->Constituents.At(a))->ET > ((Tower*) jet->Constituents.At(b))->ET; });

  double tot = 0;
  for (auto ic : ecalDau) {
    auto tower = ((Tower*) jet->Constituents.At(ic));
    if (debug) std::cout << tower->ET << " " << tower->Eta << " " << tower->Phi <<std::endl;
    tot += tower->ET;
  }
  if (debug) std::cout << tot << std::endl;

  std::vector<cluster> ecalClusters;

  // go through ecal daughters and build ecalClusters
  while (!ecalDau.empty()) {
    auto clst = (Tower*) jet->Constituents.At(ecalDau[0]);
    std::vector<int> iCl;

    // in reverse so we can easily erase later
    for (int i = ecalDau.size()-1; i > 0; --i) {
      auto cell = (Tower*) jet->Constituents.At(ecalDau[i]);
      // test within a 3x3 cell, 1 cell = 0.0174 based on CMS data card
      if ((fabs(cell->Eta - clst->Eta) < 2.5*0.0174) &&
	  (DeltaPhi(cell->Phi, clst->Phi) < 2.5*0.0174))
	iCl.push_back(i);
    }
    auto next = cluster{clst->ET, clst->Eta, clst->Phi, 0};
    for (auto i : iCl)
      next.pt += ((Tower*) jet->Constituents.At(ecalDau[i]))->ET;

    for (auto i : iCl) 
      ecalDau.erase(ecalDau.begin()+i);
    ecalDau.erase(ecalDau.begin()+0);

    ecalClusters.push_back(next);
  }


  tot = 0;
  if (debug) std::cout << std::endl << std::endl;
  for (auto cl : ecalClusters) {
    if (debug) std::cout << cl.pt << " " << cl.eta << " " << cl.phi << std::endl;
    tot += cl.pt;
  }
  if (debug) std::cout << tot << std::endl;
  if (debug) std::cout << std::endl;
  if (debug) std::cout << std::endl;
  

  std::vector<cluster> hcalClusters;
  tot = 0;
  for (auto ic : hcalDau) {
    auto tower = ((Tower*) jet->Constituents.At(ic));
    if (debug) std::cout << tower->ET << " " << tower->Eta << " " << tower->Phi <<std::endl;
    tot += tower->ET;
    hcalClusters.push_back(cluster{tower->ET, tower->Eta, tower->Phi, 1});
  }
  if (debug) std::cout << tot << std::endl;

  // // merging step
  std::vector<cluster> mergeClusters;
  for (auto ecl : ecalClusters) {
    bool foundh = false;
    for (size_t ih = 0; ih < hcalClusters.size(); ++ih) {
      if ((fabs(ecl.eta - hcalClusters[ih].eta) < 2.5*0.0174) &&
  	  DeltaPhi(ecl.phi, hcalClusters[ih].phi) < 2.5*0.0174) {
  	// ecal should be better resolved in position
  	mergeClusters.push_back(cluster{ecl.pt+hcalClusters[ih].pt, ecl.eta, ecl.phi, 1});
  	foundh = true;
  	hcalClusters.erase(hcalClusters.begin()+ih);
  	break;
      }
    }
    if (!foundh)
      mergeClusters.push_back(ecl);
  }

  // for (auto ecl : ecalClusters)
  //   mergeClusters.push_back(ecl);
  
  for (auto hcl : hcalClusters)
    mergeClusters.push_back(hcl);


  if (debug) std::cout << std::endl;
  if (debug) std::cout << std::endl;
  tot = 0;
  nmult = mergeClusters.size();
  for (auto m : mergeClusters) {
    if (debug) std::cout << m.pt << " " << m.eta << " " << m.phi <<std::endl;
    tot += m.pt;

    dau_pt.push_back(m.pt);
    dau_deta.push_back(m.eta - jet->Eta);
    dau_dphi.push_back(DeltaPhi(m.phi, jet->Phi));

    dau_ishadronic.push_back(m.ishadronic);
    dau_charge.push_back(0);

    if (m.pt > leading_dau_pt) {
      leading_dau_pt = m.pt;
      leading_dau_eta = m.eta - jet->Eta;
    }
  }
  if (debug) std::cout << tot << std::endl;

  if (debug) exit(1);
}

void fillDaughters_HADMERGE(Jet *jet, float& leading_dau_pt, float& leading_dau_eta,
			    std::vector<float> &dau_pt, std::vector<float> &dau_deta, std::vector<float> &dau_dphi,
			    std::vector<int> &dau_charge, std::vector<int> &dau_ishadronic, int& nmult)
{
  leading_dau_pt = leading_dau_eta = 0.0;
  nmult = 0;
  size_t n_dau = jet->Constituents.GetEntries();
  // fill hadronic first
  int nhad = 0;
  
  for (size_t ic = 0; ic < n_dau; ++ic) {
    double dpt, deta, dphi;
    auto dau = jet->Constituents.At(ic);
    // Constituents can be a tower (neutral) or a track (charged)
    auto tower = dynamic_cast<Tower*>(dau);
    if (!tower) continue;
    if (tower->ET < 1.0) { // Don't accept low energy neutrals
      continue;
    }

    if ((tower->Ehad != 0.0) && (tower->Eem != 0.0))
      std::cout << "ERROR: Tower with Had " << tower->Ehad << " and EM " << tower->Eem << " energy" << std::endl;

    if (tower->Ehad == 0.0) {
      continue;
    }
    dpt = tower->ET;
    deta = tower->Eta - jet->Eta;
    dphi = DeltaPhi(tower->Phi, jet->Phi);
    dau_charge.push_back(0);
    dau_ishadronic.push_back(1);
    dau_pt.push_back(dpt);
    dau_deta.push_back(deta);
    dau_dphi.push_back(dphi);
    nhad++;
    nmult++;
    if (dpt > leading_dau_pt) {
      leading_dau_pt = dpt;
      leading_dau_eta = deta;
    }
  }

  // then fill emag+track
  for (size_t ic = 0; ic < n_dau; ++ic) {
    auto dau = jet->Constituents.At(ic);

    double dpt, deta, dphi, dr;
    // Constituents can be a tower (neutral) or a track (charged)
    if (auto track = dynamic_cast<Track*>(dau)) {
      dpt = track->PT;
      deta = track->Eta - jet->Eta;
      dphi = DeltaPhi(track->Phi, jet->Phi);
      dau_charge.push_back(track->Charge);
      dau_ishadronic.push_back(0);
      dau_pt.push_back(dpt);
      dau_deta.push_back(deta);
      dau_dphi.push_back(dphi);
      if (dpt > leading_dau_pt) {
	leading_dau_pt = dpt;
	leading_dau_eta = deta;
      }
    } else if (auto tower = dynamic_cast<Tower*>(dau)) {
      if (tower->ET < 1.0) { // Don't accept low energy neutrals
	continue;
      }
      if (tower->Eem == 0.0) {
	continue;
      }

      dpt = tower->ET;
      deta = tower->Eta - jet->Eta;
      dphi = DeltaPhi(tower->Phi, jet->Phi);
	  
      bool found = false;
      for (size_t ih; ih < nhad; ++ih) {
	if ((fabs(dau_deta[ih]) < 1.653) && (fabs(deta - dau_deta[ih]) < 0.08) && (DeltaPhi(dphi, dau_dphi[ih]) < 0.08)) {
	  found = true;
	} else if ((fabs(dau_deta[ih]) < 4.35) && (fabs(deta - dau_deta[ih]) < 0.16) && (DeltaPhi(dphi, dau_dphi[ih]) < 0.16)) {
	  found = true;
	}

	// if the emag tower within the bounds, add the energy to the hadr. tower
	if (found) {
	  dau_pt[ih] += dpt;
	  if (dau_pt[ih] > leading_dau_pt) {
	    leading_dau_pt = dau_pt[ih];
	    leading_dau_eta = dau_deta[ih];
	  }
	  break;
	}
      }

      if (!found) {
	nmult++;
	dau_charge.push_back(0);
	dau_ishadronic.push_back(1);
	dau_pt.push_back(dpt);
	dau_deta.push_back(deta);
	dau_dphi.push_back(dphi);
	if (dpt > leading_dau_pt) {
	  leading_dau_pt = dpt;
	  leading_dau_eta = deta;
	}
      }
    } else {
      std::cout << "BAD DAUGHTER! " << dau << std::endl;
    }
  } 
}

void fillDaughters(
    Jet *jet,
    float& leading_dau_pt,
    float& leading_dau_eta,
	std::vector<float> &dau_pt,
    std::vector<float> &dau_deta,
    std::vector<float> &dau_dphi,
	std::vector<int> &dau_charge,
    std::vector<int> &dau_ishadronic,
    std::vector<float> &dau_eemfrac,
    std::vector<float> &dau_ehadfrac,
    std::vector<int> &dau_pid,
    std::vector<int> &dau_istrack,
    int& nmult,
    int& cmult)
{
    size_t n_dau = jet->Constituents.GetEntries();

    double dpt = 0, deta = 0, dphi = 0;
    nmult = cmult = 0;
    leading_dau_eta = leading_dau_pt = 0;

    for (size_t ic = 0; ic < n_dau; ++ic) {
        auto dau = jet->Constituents.At(ic);
        // Constituents can be a tower (neutral) or a track (charged)

        float deta = 10, dphi = 10, dr = 10, dpt = 0;


        // Neutral Hadron, photon
        if (auto tower = dynamic_cast<Tower*>(dau)) {
            dau_istrack.push_back(1);
            dau_pid.push_back(0);
            // if (tower->ET < 1.0) { // Don't accept low energy neutrals
            // 	continue;
            // }


            dpt = tower->ET;
            deta = tower->Eta - jet->Eta;
            dphi = DeltaPhi(tower->Phi, jet->Phi);

            dau_charge.push_back(0);
            nmult++;

            if (tower->Eem == 0.0)
                dau_ishadronic.push_back(1);
            else if (tower->Ehad == 0.0)
                dau_ishadronic.push_back(0);
            else
                std::cout << "ERROR: Tower with Had " << tower->Ehad << " and EM " << tower->Eem << " energy" << std::endl;

            // class Tower
            // E: calorimeter tower energy
            // Eem: calorimeter tower electromagnetic energy 
            // Ehad: calorimeter tower hadronic energy
            dau_eemfrac.push_back(tower->Eem / tower->E);
            dau_ehadfrac.push_back(tower->Ehad / tower->E);
        }
        else if (auto track = dynamic_cast<Track*>(dau)) {
            dau_istrack.push_back(0.0);
            dau_eemfrac.push_back(0.0);
            dau_ehadfrac.push_back(0.0);


            dpt = track->PT;
            deta = track->Eta - jet->Eta;
            dphi = DeltaPhi(track->Phi, jet->Phi);

            dau_charge.push_back(track->Charge);
            cmult++;

            dau_ishadronic.push_back(0);

            // class Track
            // PID: HEP ID number
            dau_pid.push_back(track->PID);

        }
        else {
          std::cout << "BAD DAUGHTER! " << dau << std::endl;
        }

        dau_pt.push_back(dpt);
        dau_deta.push_back(deta);
        dau_dphi.push_back(dphi);

        if (dpt > leading_dau_pt) {
            leading_dau_pt = dpt;
            leading_dau_eta = deta;
        }
    }
}
