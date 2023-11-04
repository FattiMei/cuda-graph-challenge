#ifndef __REACHABILITY_H_INCLUDED__
#define __REACHABILITY_H_INCLUDED__


#include <vector>
#include "graph.hpp"


std::vector<int> cpuReachability(CSRGraph &G, size_t &kernelTimeMilliseconds);
std::vector<int> gpuReachability(CSRGraph &G, size_t &kernelTimeMilliseconds);


#endif
