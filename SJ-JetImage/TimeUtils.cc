#ifndef JETIMAGE_TIMEUTILS_
#define JETIMAGE_TIMEUTILS_

#include <chrono>
#include <iostream>

typedef std::chrono::high_resolution_clock::time_point Time;
typedef std::chrono::duration<double> Duration;

class Timer {
private:
    Time start_time_, finish_time_; 
    Duration elapsed_time_;

public:
    Timer(bool init_with_start=false) {
        if( init_with_start )
            this->Start();
    }

    void Start() {
        start_time_ = std::chrono::high_resolution_clock::now();
    }

    void Finish() {
        finish_time_ = std::chrono::high_resolution_clock::now();
    }

    void Reset() {
        this->Start();
    }

    void MeasureElapsedTime()
    {
        this->Finish();
        elapsed_time_ = finish_time_ - start_time_;
    }

    double GetElapsedTime()
    {
        this->MeasureElapsedTime();
        double elapsed_time =  this->elapsed_time_.count();
        return elapsed_time;
    }

    void Print() {
        this->GetElapsedTime();
        std::cout << "Elapsed time: " << elapsed_time_.count() << " s\n";
    }

};

char* GetNow()
{
    std::chrono::time_point<std::chrono::system_clock> time_now = std::chrono::system_clock::now();
    std::time_t time_now_t = std::chrono::system_clock::to_time_t(time_now);
    char* now = std::ctime(&time_now_t);
    return now;
}


#endif // JETIMAGE_TIMEUTILS
