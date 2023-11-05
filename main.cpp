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
		<< " ms"
		<< std::endl
		<< "gpu naive "
		<< gpuTime
		<< " ms"
		<< std::endl
		<< "gpu opt "
		<< gpuOptTime
		<< " ms"
		<< std::endl
		<< std::endl;


	std::cout
		<< "gpu naive "
		<< (checkCorrectness(reference, gpuVisited) == true ? "ok" : "bad")
		<< std::endl
		<< "gpu opt "
		<< (checkCorrectness(reference, gpuOptVisited) == true ? "ok" : "bad")
		<< std::endl
		<< std::endl;



	return 0;
}
