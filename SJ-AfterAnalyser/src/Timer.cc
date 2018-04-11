#include "Timer.h"

#include <chrono>
#include <iostream>

Timer::Timer(bool init_with_start=false) {
  if( init_with_start )
    this->Start();
}

void Timer::Start() {
    start_time_ = std::chrono::high_resolution_clock::now();
}

void Timer::Finish() {
    finish_time_ = std::chrono::high_resolution_clock::now();
}

void Timer::Reset() {
    this->Start();
}

void Timer::MeasureElapsedTime()
{
    this->Finish();
    elapsed_time_ = finish_time_ - start_time_;
}

double Timer::GetElapsedTime()
{
    this->MeasureElapsedTime();
    double elapsed_time =  this->elapsed_time_.count();
    return elapsed_time;
}

void Timer::Print() {
    this->GetElapsedTime();
    std::cout << "Elapsed time: " << elapsed_time_.count() << " s\n";
}

