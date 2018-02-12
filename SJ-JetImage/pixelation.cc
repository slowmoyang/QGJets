#include <math.h>
#include <memory>

// Sqrt
class Pixelation {
private:
    Float_t eta_up_, phi_up_;
    Int_t eta_num_bins_,phi_num_bins_;

public:
    Pixelation(Float_t eta_up=0.4, Int_t eta_num_bins=33,
               Float_t phi_up=0.4, Int_t phi_num_bins=33)
    {
        eta_up_       = eta_up;
        eta_num_bins_ = eta_num_bins; 
        phi_up_       = phi_up;
        phi_num_bins_ = phi_num_bins; 
    }


    Int_t Pixelate(const Float_t& eta,
                   const Float_t& phi)
    {
        Int_t eta_idx = static_cast<Int_t>(eta_num_bins_ * (eta + eta_up_) / (2 * eta_up_));
        Int_t phi_idx = static_cast<Int_t>(phi_num_bins_ * (phi + phi_up_) / (2 * phi_up_));

        Int_t idx = eta_num_bins_ * eta_idx + phi_idx;

        return idx;
    }
};


// Sqrt
class SqrtPixelation {
private:
    Float_t eta_up_, phi_up_, eta_up_sqrt_, phi_up_sqrt_;
    Int_t eta_num_bins_,phi_num_bins_;

    Float_t Sqrt(Float_t x)
    {
        return x >= 0 ? sqrt(x) : -1 * sqrt( abs (x) );
    }


public:
    SqrtPixelation(Float_t eta_up=0.4, Int_t eta_num_bins=33,
                   Float_t phi_up=0.4, Int_t phi_num_bins=33)
    {
        eta_up_       = eta_up;
        eta_num_bins_ = eta_num_bins; 
        phi_up_       = phi_up;
        phi_num_bins_ = phi_num_bins; 

        eta_up_sqrt_ = sqrt(eta_up);
        phi_up_sqrt_ = sqrt(phi_up);
    }


    Int_t Pixelate(Float_t eta, Float_t phi)
    {
        Float_t eta_sqrt = Sqrt(eta);
        Float_t phi_sqrt = Sqrt(phi);

        Int_t eta_idx = static_cast<Int_t>(eta_num_bins_ * (eta_sqrt + eta_up_sqrt_) / (2 * eta_up_sqrt_));
        Int_t phi_idx = static_cast<Int_t>(phi_num_bins_ * (phi_sqrt + phi_up_sqrt_) / (2 * phi_up_sqrt_));

        Int_t idx = eta_num_bins_ * eta_idx + phi_idx;

        return idx;
    }
};


class SigmoidPixelation {
private:
    Float_t eta_up_, phi_up_, threshold_;
    Int_t eta_num_bins_,phi_num_bins_;

public:
    SigmoidPixelation(Float_t eta_up=0.4, Int_t eta_num_bins=33,
                      Float_t phi_up=0.4, Int_t phi_num_bins=33,
                      Float_t threshold=0.1)
    {
        eta_up_       = eta_up;
        eta_num_bins_ = eta_num_bins; 
        phi_up_       = phi_up;
        phi_num_bins_ = phi_num_bins; 

        threshold_ = threshold;
    }

    Float_t ComputeSigmoid(Float_t x)
    {
        Float_t expit = 1.0 / (1.0 + exp(-1 * x));
        return expit;
    }


    Int_t Pixelate(Float_t eta, Float_t phi)
    {
        Float_t eta_expit = ComputeSigmoid(eta/threshold_);
        Float_t phi_expit = ComputeSigmoid(phi/threshold_);

        Int_t eta_idx = static_cast<Int_t>(eta_num_bins_ * eta_expit);
        Int_t phi_idx = static_cast<Int_t>(phi_num_bins_ * phi_expit);

        Int_t idx = eta_num_bins_ * eta_idx + phi_idx;
        return idx;
    }
};

