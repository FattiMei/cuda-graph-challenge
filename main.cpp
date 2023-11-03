#include <iostream>
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


	/*
	std::vector<int> cpuVisited = cpuReachability(cudaGraph);
	std::vector<int> gpuVisited = gpuReachability(cudaGraph);


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
		<< "nodi raggiunti (CPU) "
		<< reached(cpuVisited)
		<< std::endl
		<< "nodi raggiunti (GPU) "
		<< reached(gpuVisited)
		<< std::endl;
	*/


	return 0;
}
