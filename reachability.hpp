#ifndef __REACHABILITY_H_INCLUDED__
#define __REACHABILITY_H_INCLUDED__


#include <vector>
#include "graph.hpp"


std::vector<int> cpuReachability(CSRGraph &G);
std::vector<int> gpuReachability(CSRGraph &G);


void cpuKernel(
	 int *nodePtrs
	,int *nodeNeighbors
	,int *nodeVisited
	,int *currLevelNodes
	,int *nextLevelNodes
	,const unsigned int numCurrLevelNodes
	,int *numNextLevelNodes);


#endif
