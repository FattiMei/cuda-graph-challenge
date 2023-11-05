#include <iostream>
#include <chrono>
#include "graph.hpp"
#include "parse.hpp"
#include "reachability.hpp"


bool checkCorrectness(std::vector<int> &reference, std::vector<int> &check){
	for(int i = 0; i < reference.size(); ++i){
		if(reference[i] != check[i]){
			return false;
		}
	}

	return true;
}


int main(int argc, char *argv[]){
	SimpleGraph simpleGraph = parseGraph(std::cin);
	CSRGraph cudaGraph(simpleGraph);
	CudaContext ctx(cudaGraph);


	size_t cpuTime,
		   gpuTime,
		   gpuOptTime;


	std::vector<int> reference     = cpuReachability   (cudaGraph, cpuTime);
	std::vector<int> gpuVisited    = gpuReachability   (ctx, gpuTime);
	std::vector<int> gpuOptVisited = gpuReachabilityOpt(ctx, gpuOptTime);


	std::cout
		<< "cpu reference "
		<< cpuTime
		<< " us"
		<< std::endl

		<< "gpu naive "
		<< gpuTime
		<< " us"
		<< "\t"
		<< "speedup "
		<< (double) cpuTime / (double) gpuTime
		<< std::endl

		<< "gpu opt "
		<< gpuOptTime
		<< " us"
		<< "\t"
		<< "speedup "
		<< (double) cpuTime / (double) gpuOptTime
		<< std::endl
		<< std::endl;


	if(!checkCorrectness(reference, gpuVisited)){
		std::cout << "gpu naive bad" << std::endl;
	}
	if(!checkCorrectness(reference, gpuOptVisited)){
		std::cout << "gpu opt bad" << std::endl;
	}



	return 0;
}
