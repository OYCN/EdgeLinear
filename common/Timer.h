#ifndef _TIMER_H_
#define _TIMER_H_

#ifdef WIN32
#include <windows.h>

class Timer {
private:
	__int64 freq, tStart, tStop;

public:
	Timer(){
		// Get the frequency of the hi-res timer
		QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	} //end-TimerClass

	void Start(){
		// Use hi-res timer
		QueryPerformanceCounter((LARGE_INTEGER*)&tStart);
	} //end-Start

	void Stop(){
		// Perform operations that require timing
		QueryPerformanceCounter((LARGE_INTEGER*)&tStop);
	} //end-Stop

	// Returns time in milliseconds
	double ElapsedTime(){
		// Calculate time difference in milliseconds
		return ((double)(tStop - tStart) / (double)freq)*1e3;
	} //end-Elapsed
};
#elif defined LINUX

#include <chrono>
#include <stdio.h>
typedef std::chrono::high_resolution_clock Clock;
#ifdef CYCLING
#define TDEF(x_) static unsigned long long int x_##_t0, x_##_t1;
#define TSTART(x_) x_##_t0 = __rdtsc();
#define TEND(x_) x_##_t1 = __rdtsc();
#define TPRINT(x_, str) printf("%-20s \t%.6f\t M cycles\n", str, (double)(x_##_t1 - x_##_t0)/1e6);
#define TPRINTMS(x_, str) printf("%-20s \t%.6f\t M cycles\n", str, (double)(x_##_t1 - x_##_t0)/1e6);
#define TPRINTUS(x_, str) printf("%-20s \t%.6f\t M cycles\n", str, (double)(x_##_t1 - x_##_t0)/1e6);
#elif defined TIMING
#define TDEF(x_) std::chrono::high_resolution_clock::time_point x_##_t0, x_##_t1;
#define TSTART(x_) x_##_t0 = Clock::now();
#define TEND(x_) x_##_t1 = Clock::now();
#define TPRINT(x_, str) printf("%-20s \t%.6f\t sec\n", str, std::chrono::duration_cast<std::chrono::microseconds>(x_##_t1 - x_##_t0).count()/1e6);
#define TPRINTMS(x_, str) printf("%-20s \t%.6f\t ms\n", str, std::chrono::duration_cast<std::chrono::microseconds>(x_##_t1 - x_##_t0).count()/1e3);
#define TPRINTUS(x_, str) printf("%-20s \t%ld\t us\n", str, std::chrono::duration_cast<std::chrono::microseconds>(x_##_t1 - x_##_t0).count());
#else
#define TDEF(x_)
#define TSTART(x_)
#define TEND(x_)
#define TPRINT(x_, str)
#define TPRINTMS(x_, str)
#define TPRINTUS(x_, str)
#endif

#endif

#endif