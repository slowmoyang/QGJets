#ifndef TIMER_H_
#define TIMER_H_

#include <chrono>
#include <iostream>

typedef std::chrono::high_resolution_clock::time_point Time;
typedef std::chrono::duration<double> Duration;

class Timer {
private:
    Time start_time_, finish_time_; 
    Duration elapsed_time_;

public:
    Timer(bool init_with_start);

    void Start();
    void Finish();
    void Reset();
    void MeasureElapsedTime();
    double GetElapsedTime();
    void Print();
};

#endif // JETIMAGE_TIMEUTILS
