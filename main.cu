#include <iostream>
#include <cuda.h>
#include "graph.hpp"
#include "parse.hpp"
#include "reachability.hpp"


int main(int argc, char *argv[]){
	SimpleGraph simpleGraph = parseGraph(std::cin);
	CudaGraph cudaGraph(simpleGraph);


	int *currLevelNodes = new int[cudaGraph.nodeCount];
	int *nextLevelNodes = new int[cudaGraph.nodeCount];
	int *nodeVisited    = new int[cudaGraph.nodeCount];
	int numCurrLevelNodes;
	int numNextLevelNodes;


	// inizializzazioni dei nodi visitati
	for(int i = 0; i < cudaGraph.nodeCount; ++i)
		nodeVisited[i] = 0;


	// inizializzazione della coda
	numCurrLevelNodes = 1;
	currLevelNodes[0] = 0;


	while(numCurrLevelNodes != 0){
		numNextLevelNodes = 0;

		cpuKernel(
				cudaGraph.nodePtrs
				,cudaGraph.nodeNeighbors
				,nodeVisited
				,currLevelNodes
				,nextLevelNodes
				,numCurrLevelNodes
				,&numNextLevelNodes
				);

		numCurrLevelNodes = numNextLevelNodes;
		std::swap(currLevelNodes, nextLevelNodes);
	}


	delete[] currLevelNodes;
	delete[] nextLevelNodes;
	delete[] nodeVisited;


	return 0;
}
