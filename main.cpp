#include <iostream>
#include <cuda.h>
#include "graph.hpp"
#include "parse.hpp"
#include "reachability.hpp"


int main(int argc, char *argv[]){
	SimpleGraph simpleGraph = parseGraph(std::cin);
	CudaGraph cudaGraph(simpleGraph);


	std::vector<int> cpuVisited = cpuReachability(cudaGraph);
	// std::vector<int> gpuVisited = gpuReachability(cudaGraph);
	

	int sum = 0;
	for(int p : cpuVisited)
		sum += p;


	std::cout << "raggiunti " << sum << " nodi" << std::endl;




	return 0;
}
