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


	size_t cpuTotalTime,
		   cpuComputationTime,
		   gpuTotalTime,
		   gpuComputationTime;


	// misuro i tempi totali di computazione (trasferimento incluso)
	using namespace std::chrono;
	std::chrono::high_resolution_clock clock;


	auto t0 = clock.now();
		std::vector<int> cpuVisited = cpuReachability(cudaGraph, cpuComputationTime);
	auto t1 = clock.now();

	cpuTotalTime = duration_cast<milliseconds>(t1-t0).count();



	t0 = clock.now();
		std::vector<int> gpuVisited = gpuReachability(cudaGraph, gpuComputationTime);
	t1 = clock.now();

	gpuTotalTime = duration_cast<milliseconds>(t1-t0).count();



	bool correct = true;
	for(int i = 0; i < cpuVisited.size(); ++i){
		if(cpuVisited[i] != gpuVisited[i]){
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
		<< "(CPU) raggiunti "
		<< reached(cpuVisited)
		<< " nodi in "
		<< cpuTotalTime
		<< " ms"
		<< "\t"
		<< "computation time "
		<< cpuComputationTime
		<< " ms"
		<< std::endl
		<< "(GPU) raggiunti "
		<< reached(gpuVisited)
		<< " nodi in "
		<< gpuTotalTime
		<< " ms"
		<< "\t"
		<< "computation time "
		<< gpuComputationTime
		<< " ms"
		<< std::endl;


	return 0;
}
