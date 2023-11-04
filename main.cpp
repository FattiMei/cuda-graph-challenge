#include <iostream>
#include <chrono>
#include <cuda.h>
#include "graph.hpp"
#include "parse.hpp"
#include "reachability.hpp"


int reached(std::vector<int> const &V){
	int result = 0;


	for(int p : V){
		if(p > 0)
			++result;
	}


	return result;
}


int main(int argc, char *argv[]){
	SimpleGraph simpleGraph = parseGraph(std::cin);
	CSRGraph cudaGraph(simpleGraph);


	// misuro i tempi totali di computazione (trasferimento incluso)
	using namespace std::chrono;
	std::chrono::high_resolution_clock clock;


	size_t gpuTime,
		   gpuOptTime;


	std::vector<int> gpuVisited = gpuReachability(cudaGraph, gpuTime);
	std::vector<int> gpuOptVisited = gpuReachabilityOptimized(cudaGraph, gpuOptTime);



	bool correct = true;
	for(int i = 0; i < gpuVisited.size(); ++i){
		if(gpuVisited[i] != gpuOptVisited[i]){
			correct = false;
			break;
		}
	}


	if(correct){
		std::cout << "implementazione gpu corretta" << std::endl;
	}
	else{
		std::cout << "implementazione gpu errata" << std::endl;
	}


	std::cout
		<< "gpu naive "
		<< gpuTime
		<< " ms"
		<< std::endl
		<< "gpu opt "
		<< gpuOptTime
		<< " ms"
		<< std::endl;


	return 0;
}
