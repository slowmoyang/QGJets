#include "TFile.h"
#include "TTree.h"
#include "TClonesArray.h"
#include "TMath.h"
#include "TString.h"
#include "TSystem.h"
#include "TRefArray.h"
#include "TMatrixDfwd.h"
#include "TVectorD.h"

#include "classes/DelphesClasses.h"

#include <iostream>
#include <string>
#include <memory>
#include <vector>

using std::string;

const Double_t kMinJetPT = 20.0;
const Double_t kMaxJetEta = 2.4;
const Double_t kdRCut = 0.3;
bool doHadMerge = false;
bool doEcalCl = false;
bool doEtaFlip = false;

bool debug=false;


Double_t computeDeltaPhi(Double_t phi1, Double_t phi2)
{
  static const Double_t kPI = TMath::Pi();
  static const Double_t kTWOPI = 2*TMath::Pi();
  Double_t x = phi1 - phi2;
  if(TMath::IsNaN(x))
  {
    std::cerr << "computeDeltaPhi function called with NaN" << std::endl;
    return x;
  }
  while (x >= kPI) x -= kTWOPI;
  while (x < -kPI) x += kTWOPI;
  return x;
}


Double_t computeDeltaEta(Double_t constituent_eta, Double_t jet_eta, Bool_t use_flip=false)
{
  Double_t deta = constituent_eta - jet_eta;

  if(use_flip and (jet_eta < 0))
  {
      deta *= -1;
  }

  return deta;
}


Double_t computeDeltaR(Double_t deta, Double_t dphi)
{
  return std::sqrt(std::pow(deta, 2) + std::pow(dphi, 2));
}


std::tuple<Double_t, Double_t> computeAxes(const std::vector<double> & dau_deta,
                                           const std::vector<double> & dau_dphi,
                                           const std::vector<double> & dau_pt);


bool isBalanced(TClonesArray * gen_jets);
bool passZjets(TClonesArray * jets, TClonesArray * muons, TClonesArray * electrons, int &nGoodJets);


void fillDaughters(Jet *jet,
                   float &leading_dau_pt,
                   float& leading_dau_eta,
                   std::vector<float> &dau_pt,
                   std::vector<float> &dau_deta,
                   std::vector<float> &dau_dphi,
                   std::vector<float> &dau_eta,
                   std::vector<float> &dau_phi,
                   std::vector<int> &dau_charge,
                   std::vector<int> &dau_ishadronic,
                   std::vector<float> &dau_eemfrac,
                   std::vector<float> &dau_ehadfrac,
                   std::vector<int> &dau_pid,
                   std::vector<int> &dau_istrack,
                   TClonesArray &dau_p4,
                   int& nmult,
                   int& cmult);


int main(int argc, char *argv[])
{
  // Argument Parsing

  int c;
  while ((c = getopt(argc, argv, "ehf")) != -1) {
    switch (c)
    {
      case 'f':
        doEtaFlip = true;
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

  auto in_path = string{argv[optind]};
  auto out_path = string{argv[optind+1]};

  std::cout << "Processing '" << in_path << "' into '" << out_path << "'" << std::endl;

  auto in_file = TFile::Open(in_path.c_str());
  auto in_tree = dynamic_cast<TTree*>(in_file->Get("Delphes"));
  in_tree->SetBranchStatus("*", true);

  
  TClonesArray *jets = 0;
  in_tree->SetBranchAddress("Jet", &jets);
  TClonesArray *gen_jets = 0;
  in_tree->SetBranchAddress("GenJet", &gen_jets);
  TClonesArray *particles = 0;
  in_tree->SetBranchAddress("Particle", &particles);

  TClonesArray *electrons = 0;
  in_tree->SetBranchAddress("Electron", &electrons);
  TClonesArray *muons = 0;
  in_tree->SetBranchAddress("Muon", &muons);

  TClonesArray *vertices = 0;
  in_tree->SetBranchAddress("Vertex", &vertices);

  TClonesArray *eflow_tracks = 0;
  in_tree->SetBranchAddress("EFlowTrack", &eflow_tracks);
  TClonesArray *eflow_photons = 0;
  in_tree->SetBranchAddress("EFlowPhoton", &eflow_photons);
  TClonesArray *eflow_nhads = 0;
  in_tree->SetBranchAddress("EFlowNeutralHadron", &eflow_nhads);
 
 
  auto out_file = TFile::Open(out_path.c_str(), "RECREATE");
  auto out_tree = new TTree{"jetAnalyser", "jetAnalyser"};

  // Matching Jason's jetAnalyser output
  #define Branch_(type, name, suffix) type name = 0; out_tree->Branch(#name, &name, #name "/" #suffix);
  #define BranchI(name) Branch_(Int_t, name, I)
  #define BranchF(name) Branch_(Float_t, name, F)
  #define BranchO(name) Branch_(Bool_t, name, O)
  #define BranchA_(type, name, size, suffix) type name[size] = {0.}; out_tree->Branch(#name, &name, #name"["#size"]/"#suffix);
  #define BranchAI(name, size) BranchA_(Int_t, name, size, I);
  #define BranchAF(name, size) BranchA_(Float_t, name, size, F);
  #define BranchAO(name, size) BranchA_(Bool_t, name, size, O);
  #define BranchVF(name) std::vector<float> name; out_tree->Branch(#name, "vector<float>", &name);
  #define BranchVI(name) std::vector<int> name; out_tree->Branch(#name, "vector<int>", &name);
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
  BranchF(major_axis);
  BranchF(minor_axis);
  BranchF(average_width);
  BranchF(eccentricity);
  BranchF(leading_dau_pt);
  BranchF(leading_dau_eta);
  BranchI(nmult);
  BranchI(cmult);
  BranchI(partonId);
  BranchI(flavorId);

  BranchI(flavorAlgoId);
  BranchI(flavorPhysId);

  BranchVF(dau_pt);
  BranchVF(dau_deta);
  BranchVF(dau_dphi);
  BranchVF(dau_eta);
  BranchVF(dau_phi);
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

  /// FIXME Use std::vector
  TClonesArray dau_p4("TLorentzVector");
  out_tree->Branch("dau_p4", &dau_p4, 256000, 0);
  // FIXME 
  // dau_p4.BypassStreamer();

  // std::vector<TLorentzVector> dau_p4;
  // out_tree->Branch("dau_p4", "vector<TLorentzVector>", &dau_p4);

  // Satellites
  TClonesArray satellites_p4("TLorentzVector");
  out_tree->Branch("satellites_p4", &satellites_p4, 256000, 0);
  // satellites_p4.BypassStreamer();

  bool firstTime = true, badHardGenSeen = false;


  const int kNumEvents = in_tree->GetEntries();
  const TString kPrintFmt = TString::Format("[%s/%d]", "%d", kNumEvents);
  for (size_t iev = 0; iev < in_tree->GetEntries(); ++iev)
  {
    if( iev % 1000 == 0)
    {
      std::cout << TString::Format(kPrintFmt, iev) << std::endl;
    }

    in_tree->GetEntry(iev);
    nEvent = iev;
    nJets = jets->GetEntries();
    order = 0;

    nPriVtxs = vertices ? vertices->GetEntries() : 1;
    
    pass_Zjets = passZjets(jets, muons, electrons, nGoodJets);

    // GenParticle 
    std::vector<const GenParticle*> hardGen;
    for (unsigned k = 0; k < particles->GetEntries(); ++k)
    {
      auto p = static_cast<const GenParticle *>(particles->At(k));

      if (p->Status != 23) continue; // Status 23 is hard process parton in Pythia8
      //if (p->Status < 20 || p->Status > 29) continue; // All 20s are hard processes (not all of them make sense but this is how QGL does it)
      if (std::abs(p->PID) > 5 and p->PID != 21) continue; // consider light quarks and gluons only

      hardGen.push_back(p);
 
      if (firstTime)
      {
	    std::cout << "WARNING: ASSUMING PYTHIA8 HARDQCD GENERATION, ONLY 2 HARD PARTONS CONSIDERED" << std::endl;
	    firstTime = false;
      }

      if (hardGen.size() == 2) break;
    }

    if (not badHardGenSeen and (hardGen.size() != 2))
    {
      std::cout << "hardGen " << hardGen.size() << std::endl;
      badHardGenSeen = true;
    }

    balanced = isBalanced(jets);

    // Satellite particles
    std::set<TObject*> satellites;
    for(unsigned i = 0; i < eflow_tracks->GetEntries(); ++i)
    {
      satellites.insert(dynamic_cast<TObject*>(eflow_tracks->At(i)));
    }

    for(unsigned i = 0; i < eflow_nhads->GetEntries(); ++i)
    {
      satellites.insert(dynamic_cast<TObject*>(eflow_nhads->At(i)));
    }

    for(unsigned i = 0; i < eflow_photons->GetEntries(); ++i)
    {
      satellites.insert(dynamic_cast<TObject*>(eflow_photons->At(i)));
    }

    for (unsigned j = 0; j < jets->GetEntries(); ++j)
    {
      auto jet = dynamic_cast<Jet*>(jets->At(j));
      for (size_t ic = 0; ic < jet->Constituents.GetEntries(); ++ic)
      {
        auto dau = jet->Constituents.At(ic);
        auto iter = satellites.find(dau);
        bool is_constituent = (iter != satellites.end());
        if(is_constituent) satellites.erase(iter);
      }
    }

    // satellites loop
    unsigned i_sat = 0;
    satellites_p4.Clear();
    for(const auto & each : satellites)
    {
      if(auto track = dynamic_cast<Track*>(each))
      {
        new(satellites_p4[i_sat]) TLorentzVector(track->P4());
      }
      else if(auto tower = dynamic_cast<Tower*>(each))
      {
        new(satellites_p4[i_sat]) TLorentzVector(tower->P4());
      }
      else
      {
        std::cout << "BAD SATELLITES PARTICLE" << std::endl;
      }
      i_sat++; 
    }  

    // JET
    for (unsigned j = 0; j < jets->GetEntries(); ++j)
    {
      lepton_overlap = false;
      if (j >= 2) balanced = false; // only top 2 balanced
      if (j >= 1) pass_Zjets = false; // only top 1 passes ZJet!

      auto jet = dynamic_cast<Jet*>(jets->At(j));

      // some cuts, check pt
      if (jet->PT < kMinJetPT) continue;
      if (std::fabs(jet->Eta) > kMaxJetEta) continue;

      // match to hard process in pythia
      matched = false;
      const GenParticle *match = 0;
      float dRMin = 10.;
      for (auto& p : hardGen)
      {
      	float dR = computeDeltaR(jet->Eta - p->Eta, computeDeltaPhi(jet->Phi, p->Phi));
      	if (dR < dRMin and dR < kdRCut)
        {
      	  matched = true;
      	  match = p;
      	}
      }
      
      // check overlapping jets
      bool overlap = false;
      for (unsigned k = 0; k < jets->GetEntries(); ++k)
      {
    	if (k == j) continue;
	    auto otherJet = dynamic_cast<Jet*>(jets->At(k));
    	float dR = computeDeltaR(jet->Eta - otherJet->Eta, computeDeltaPhi(jet->Phi, otherJet->Phi));
    	if (dR < kdRCut) overlap = true;
      }

      if (overlap) continue;

      // check overlap with lepton
      for (unsigned k = 0; k < electrons->GetEntries(); ++k)
      {
    	auto ele = dynamic_cast<Electron*>(electrons->At(k));
	    float dR = computeDeltaR(jet->Eta - ele->Eta, computeDeltaPhi(jet->Phi, ele->Phi));
    	if (dR < kdRCut) lepton_overlap = true;
      }
      for (unsigned k = 0; k < muons->GetEntries(); ++k)
      {
    	auto mu = dynamic_cast<Muon*>(muons->At(k));
	    float dR = computeDeltaR(jet->Eta - mu->Eta, computeDeltaPhi(jet->Phi, mu->Phi));
    	if (dR < kdRCut) lepton_overlap = true;
      }




      pt = jet->PT;
      eta = jet->Eta;
      phi = jet->Phi;
      nmult = jet->NNeutrals;
      cmult = jet->NCharged;

      partonId = match ? match->PID : 0;

      flavorId = jet->Flavor;
      flavorAlgoId = jet->FlavorAlgo;
      flavorPhysId = jet->FlavorPhys;

      axis1 = 0.0f;
      axis2 = 0.0f;
      major_axis = 0.0f;
      minor_axis = 0.0f;
      average_width = 0.0f;
      eccentricity = 0.0f;
      ptD = 0;
      pt_dr_log = 0;
      dau_pt.clear();
      dau_deta.clear();
      dau_dphi.clear();
      dau_eta.clear();
      dau_phi.clear();
      dau_charge.clear();
      dau_ishadronic.clear();
      dau_eemfrac.clear();
      dau_ehadfrac.clear();
      dau_pid.clear();
      dau_istrack.clear();
      dau_p4.Clear();
      test.clear();
      


      fillDaughters(jet,
                    leading_dau_pt,
                    leading_dau_eta,
                    dau_pt,
                    dau_deta,
                    dau_dphi,
                    dau_eta,
                    dau_phi,
                    dau_charge,
                    dau_ishadronic,
                    dau_eemfrac,
                    dau_ehadfrac,
                    dau_pid,
                    dau_istrack,
                    dau_p4,
                    nmult,
                    cmult);


      float pt_squared_sum = 0;
      float sum_pt = 0;
      float sum_detadphi = 0;
      float sum_deta = 0, sum_deta2 = 0, ave_deta = 0, ave_deta2 = 0;
      float sum_dphi = 0, sum_dphi2 = 0, ave_dphi = 0, ave_dphi2 = 0;

      float pt_squared = 0;
      float pt_suqared_sum = 0;
      float m00 = 0, m01 = 0, m11 = 0;
      for (size_t ic = 0; ic < dau_pt.size(); ++ic)
      {
    	double d_pt = dau_pt[ic];
	    double deta = dau_deta[ic];
    	double dphi = dau_dphi[ic];

	    pt_squared = std::pow(d_pt, 2);
        pt_squared_sum += pt_squared;

     	double dr = computeDeltaR(deta, dphi);
	
	    dr = computeDeltaR(deta, dphi);
        pt_dr_log += std::log(d_pt / dr);

        sum_pt += d_pt;
        sum_deta += deta * pt_squared;
        sum_deta2 += deta * deta * pt_squared;
        sum_dphi += dphi * pt_squared;
        sum_dphi2 += dphi*dphi* pt_squared;
        sum_detadphi += deta*dphi* pt_squared;

        m00 += pt_squared * std::pow(deta, 2);
        m11 += pt_squared * std::pow(dphi, 2);
        m01 -= pt_squared * std::fabs(deta) * std::fabs(dphi); 
      }

      float a = 0, b = 0, c = 0;
      // CMS PAS JME-13-002
      if (pt_squared_sum > 0)
      {
        ptD = std::sqrt(pt_squared_sum) / sum_pt;

        ave_deta = sum_deta / pt_squared_sum;
        ave_deta2 = sum_deta2 / pt_squared_sum;
        ave_dphi = sum_dphi / pt_squared_sum;
        ave_dphi2 = sum_dphi2 / pt_squared_sum;

        a = ave_deta2 - ave_deta * ave_deta;
        b = ave_dphi2 - ave_dphi * ave_dphi;
        c = -(sum_detadphi/pt_squared_sum - ave_deta*ave_dphi);

        float delta = sqrt(fabs((a-b)*(a-b) + 4*c*c));
        axis1 = (a+b+delta > 0) ? sqrt(0.5*(a+b+delta)) : 0;
        axis2 = (a+b-delta > 0) ? sqrt(0.5*(a+b-delta)) : 0;
      }

      TMatrixDSym jet_shape_matrix(2);
      jet_shape_matrix(0, 0) = m00;
      jet_shape_matrix(0, 1) = m01;
      jet_shape_matrix(1, 1) = m11;
      TVectorD jet_shape_eigenvalues;
      jet_shape_matrix.EigenVectors(jet_shape_eigenvalues);
      float lambda1 = jet_shape_eigenvalues[0];
      float lambda2 = jet_shape_eigenvalues[1];

      // sigma
      float s1_squared = lambda1 / pt_squared_sum;
      float s2_squared = lambda2 / pt_squared_sum;
      major_axis = std::sqrt(s1_squared);
      minor_axis = std::sqrt(s2_squared);
      if(major_axis < minor_axis)
      {
        std::cout << "Major axis < minor axis" << std::endl;
      }
      average_width = std::sqrt(s1_squared + s2_squared);
      eccentricity = std::sqrt(1 - (s2_squared / s1_squared));

      // axis1 = -std::log(axis1);
      // axis2 = -std::log(axis2);

      n_dau = dau_pt.size();

      order++;
      out_tree->Fill();
    } // END JET LOOP




  } // END EVENT LOOP

 
  out_tree->Write();
  out_file->Close();
  
  return 0;
}


/* Is the event balanced according to the criteria of pg 13 of http://cds.cern.ch/record/2256875/files/JME-16-003-pas.pdf */
bool isBalanced(TClonesArray * jets)
{
  if (jets->GetEntries() < 2) return false;

  auto obj1 = dynamic_cast<Jet*>(jets->At(0));
  auto obj2 = dynamic_cast<Jet*>(jets->At(1));

  // 2 jets of 30 GeV
  if (obj1->PT < 30.0) return false;
  if (obj2->PT < 30.0) return false;
  // that are back-to-back
  if (obj1->P4().DeltaPhi(obj2->P4()) < 2.5) return false;

  // and any 3rd jet requires pt < 30% of the avg. of the 2 leading jets
  if (jets->GetEntries() > 2)
  {
    auto obj3 = dynamic_cast<Jet*>(jets->At(2));
    return (obj3->PT < 0.3*(0.5*(obj2->PT  + obj1->PT)));
  }
  else
    return true;
}


/* Does the event pass the Zjets criteria according to the criteria of pg 11-12 of http://cds.cern.ch/record/2256875/files/JME-16-003-pas.pdf */
bool passZjets(TClonesArray * jets,
               TClonesArray * muons,
               TClonesArray * electrons,
               int & nGoodJets)
{
  bool pass_Zjets = false;

  if (jets->GetEntries() < 1) return false;

  nGoodJets = 0;
  int iMaxPt = -1;
  for (unsigned k = 0; k < jets->GetEntries(); ++k)
  {
    auto j = dynamic_cast<const Jet *>(jets->At(k));
    if (j->PT < kMinJetPT) continue;
    if (std::fabs(j->Eta) > kMaxJetEta) continue;
    if (iMaxPt < 0) iMaxPt = k;
    if ((dynamic_cast<const Jet*>(jets->At(iMaxPt)))->PT < j->PT)
      iMaxPt = k;
    
    nGoodJets++;
  }
  if (iMaxPt < 0) return false;

  // check for Z event
  TLorentzVector theDimuon;
  for (unsigned k = 0; k < muons->GetEntries(); ++k)
  {
    if (iMaxPt < 0) break;
    auto mu = dynamic_cast<Muon*>(muons->At(k));
    if (mu->PT < 20.) continue;
    for (unsigned kk = k; kk < muons->GetEntries(); ++kk)
    {
      auto mu2 = dynamic_cast<Muon*>(muons->At(kk));
      if (mu2->PT < 20.) continue;
      if (mu->Charge*mu2->Charge > 0) continue;
      auto dimuon = (mu->P4() + mu2->P4());
      if (dimuon.M() < 70. or dimuon.M() > 110.) continue;
      pass_Zjets = true;
      theDimuon = dimuon;
    }
  }

  if (pass_Zjets)
  {
    auto j = dynamic_cast<const Jet *>(jets->At(iMaxPt));
    // require them to be back to back
    if (computeDeltaPhi(j->Phi, theDimuon.Phi()) < 2.1)
      pass_Zjets = false;
    for (unsigned k = 0; k < jets->GetEntries(); ++k) {
      if (k == iMaxPt) continue;
      auto j = dynamic_cast<const Jet *>(jets->At(k));
      if (j->PT > 0.3*theDimuon.Pt())
	pass_Zjets = false;
    }
  }

  return pass_Zjets;
}


std::tuple<Double_t, Double_t> computeAxes(const std::vector<double> & dau_deta,
                                           const std::vector<double> & dau_dphi,
                                           const std::vector<double> & dau_pt)
{
  Double_t pt_squared = 0.0, pt_squared_sum = 0.0;
  Double_t m00 = 0.0, m01 = 0.0, m11 = 0.0;
  for (size_t ic = 0; ic < dau_pt.size(); ++ic)
  {
    double deta = dau_deta[ic];
    double dphi = dau_dphi[ic];

    pt_squared = std::pow(dau_pt[ic], 2);
    pt_squared_sum += pt_squared;

    m00 += pt_squared * std::pow(deta, 2);
    m11 += pt_squared * std::pow(dphi, 2);
    m01 -= pt_squared * std::fabs(deta) * std::fabs(dphi); 
  }

  TMatrixDSym jet_shape_matrix(2);
  jet_shape_matrix(0, 0) = m00;
  jet_shape_matrix(0, 1) = m01;
  jet_shape_matrix(1, 1) = m11;

  TVectorD eigenvalues;
  jet_shape_matrix.EigenVectors(eigenvalues);

  // sigma
  // float s1_squared = lambda1 / pt_squared_sum;
  // float s2_squared = lambda2 / pt_squared_sum;
  // major_axis = std::sqrt(s1_squared);
  // minor_axis = std::sqrt(s2_squared);
  // if(major_axis < minor_axis)
  // {
  //  std::cout << "Major axis < minor axis" << std::endl;
  // }
  // average_width = std::sqrt(s1_squared + s2_squared);
  // eccentricity = std::sqrt(1 - (s2_squared / s1_squared));
  Double_t major_axis = std::sqrt(eigenvalues[0] / pt_squared_sum);
  Double_t minor_axis = std::sqrt(eigenvalues[1] / pt_squared_sum);

  return std::make_tuple(major_axis, minor_axis);
}



void fillDaughters(Jet *jet,
                   float& leading_dau_pt,
                   float& leading_dau_eta,
                   std::vector<float> &dau_pt,
                   std::vector<float> &dau_deta,
                   std::vector<float> &dau_dphi,
                   std::vector<float> &dau_eta,
                   std::vector<float> &dau_phi,
                   std::vector<int> &dau_charge,
                   std::vector<int> &dau_ishadronic,
                   std::vector<float> &dau_eemfrac,
                   std::vector<float> &dau_ehadfrac,
                   std::vector<int> &dau_pid,
                   std::vector<int> &dau_istrack,
                   TClonesArray &dau_p4,
                   int& nmult,
                   int& cmult)
{

  size_t n_dau = jet->Constituents.GetEntries();

  nmult = cmult = 0;
  leading_dau_eta = leading_dau_pt = 0;

  for (size_t ic = 0; ic < n_dau; ++ic)
  {
    auto dau = jet->Constituents.At(ic);
     

      // Constituents can be a tower (neutral) or a track (charged)
    float deta = 10, dphi = 10, dr = 10, dpt = 0;
    float eta = 0, phi = 0;

    // Neutral Hadron, photon
    if (auto tower = dynamic_cast<Tower*>(dau))
    {


      dau_istrack.push_back(0);
      // if (tower->ET < 1.0) { // Don't accept low energy neutrals
      // 	continue;
      // }

      // E_T = \sqrt{ m^2 + p_T^2 }
      dpt = tower->ET;

      deta = computeDeltaEta(tower->Eta, jet->Eta, doEtaFlip);
      dphi = computeDeltaPhi(tower->Phi, jet->Phi);
      dau_eta.push_back(tower->Eta);
      dau_phi.push_back(tower->Phi);
      dau_charge.push_back(0);
      nmult++;

      if (tower->Eem == 0.0)
      {
          // hadronic
          dau_ishadronic.push_back(1);
          dau_pid.push_back(0); // Neutral hadron
      }
      else if (tower->Ehad == 0.0)
      {
          dau_ishadronic.push_back(0); // leptonic
          // stable neutral lepton --> photon
          dau_pid.push_back(22);
      }
      else
      {
          std::cout << "ERROR: Tower with Had " << tower->Ehad
                    << " and EM " << tower->Eem << " energy" << std::endl;
      }    

      // class Tower
      // E: calorimeter tower energy
      // Eem: calorimeter tower electromagnetic energy 
      // Ehad: calorimeter tower hadronic energy
      dau_eemfrac.push_back(tower->Eem / tower->E);
      dau_ehadfrac.push_back(tower->Ehad / tower->E);

      new(dau_p4[ic]) TLorentzVector(tower->P4());
    }
    else if (auto track = dynamic_cast<Track*>(dau))
    {
      dau_istrack.push_back(1);
      dau_eemfrac.push_back(0.0);
      dau_ehadfrac.push_back(0.0);

      dpt = track->PT;
      deta = computeDeltaEta(track->Eta, jet->Eta, doEtaFlip);
      dphi = computeDeltaPhi(track->Phi, jet->Phi);

      dau_eta.push_back(track->Eta);
      dau_phi.push_back(track->Phi);
      dau_charge.push_back(track->Charge);
      cmult++;

      dau_ishadronic.push_back(0);

      dau_pid.push_back(track->PID);

      new(dau_p4[ic]) TLorentzVector(track->P4());
    }
    else
    {
      std::cout << "BAD DAUGHTER! " << dau << std::endl;
    }

    dau_pt.push_back(dpt);
    dau_deta.push_back(deta);
    dau_dphi.push_back(dphi);

    if (dpt > leading_dau_pt)
    {
        leading_dau_pt = dpt;
        leading_dau_eta = deta;
    }
  }
}
