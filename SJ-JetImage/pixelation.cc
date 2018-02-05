#include <math.h>
#include <memory>

// Sqrt
class Pixelation {
private:
    float eta_up_, phi_up_;
    int eta_num_bins_,phi_num_bins_;

public:
    Pixelation(float eta_up=0.4, int eta_num_bins=33,
               float phi_up=0.4, int phi_num_bins=33)
    {
        eta_up_       = eta_up;
        eta_num_bins_ = eta_num_bins; 
        phi_up_       = phi_up;
        phi_num_bins_ = phi_num_bins; 
    }


    int Pixelate(float eta, float phi)
    {
        int eta_idx = static_cast<int>(eta_num_bins_ * (eta + eta_up_) / (2 * eta_up_));
        int phi_idx = static_cast<int>(phi_num_bins_ * (phi + phi_up_) / (2 * phi_up_));

        int idx = eta_num_bins_ * eta_idx + phi_idx;

        return idx;
    }
};


// Sqrt
class SqrtPixelation {
private:
    float eta_up_, phi_up_, eta_up_sqrt_, phi_up_sqrt_;
    int eta_num_bins_,phi_num_bins_;

    float Sqrt(float x)
    {
        return x >= 0 ? sqrt(x) : -1 * sqrt( abs (x) );
    }


public:
    SqrtPixelation(float eta_up=0.4, int eta_num_bins=33,
                   float phi_up=0.4, int phi_num_bins=33)
    {
        eta_up_       = eta_up;
        eta_num_bins_ = eta_num_bins; 
        phi_up_       = phi_up;
        phi_num_bins_ = phi_num_bins; 

        eta_up_sqrt_ = sqrt(eta_up);
        phi_up_sqrt_ = sqrt(phi_up);
    }


    int Pixelate(float eta, float phi)
    {
        float eta_sqrt = Sqrt(eta);
        float phi_sqrt = Sqrt(phi);

        int eta_idx = static_cast<int>(eta_num_bins_ * (eta_sqrt + eta_up_sqrt_) / (2 * eta_up_sqrt_));
        int phi_idx = static_cast<int>(phi_num_bins_ * (phi_sqrt + phi_up_sqrt_) / (2 * phi_up_sqrt_));

        int idx = eta_num_bins_ * eta_idx + phi_idx;

        return idx;
    }
};


class SigmoidPixelation {
private:
    float eta_up_, phi_up_, threshold_;
    int eta_num_bins_,phi_num_bins_;

public:
    SigmoidPixelation(float eta_up=0.4, int eta_num_bins=33,
                      float phi_up=0.4, int phi_num_bins=33,
                      float threshold=0.1)
    {
        eta_up_       = eta_up;
        eta_num_bins_ = eta_num_bins; 
        phi_up_       = phi_up;
        phi_num_bins_ = phi_num_bins; 

        threshold_ = threshold;
    }

    float ComputeSigmoid(float x)
    {
        float expit = 1.0 / (1.0 + exp(-1 * x));
        return expit;
    }


    int Pixelate(float eta, float phi)
    {
        float eta_expit = ComputeSigmoid(eta/threshold_);
        float phi_expit = ComputeSigmoid(phi/threshold_);

        int eta_idx = static_cast<int>(eta_num_bins_ * eta_expit);
        int phi_idx = static_cast<int>(phi_num_bins_ * phi_expit);

        int idx = eta_num_bins_ * eta_idx + phi_idx;
        return idx;
    }
};

