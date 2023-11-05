#ifndef __REACHABILITY_H_INCLUDED__
#define __REACHABILITY_H_INCLUDED__


#include <vector>
#include "graph.hpp"
#include "error.hpp"


struct CudaContext{
	int *nodePtrs;
	int *nodeNeighbors;
	int *nodeVisited;
	int *currLevelNodes;
	int *nextLevelNodes;
	int *numNextLevelNodes;
	int numCurrLevelNodes;
	int nodeCount;

	/*
		Nome				Tipo				Relazione
		nodePtrs			int[nodeCount+1]	HOST -> DEVICE
		nodeNeighbors		int[edgeCount]		HOST -> DEVICE
		nodeVisited			int[nodeCount]		DEVICE -> HOST (ma ha bisogno di essere inizializzato)
		currLevelNodes		int[nodeCount]		DEVICE
		nextLevelNodes		int[nodeCount]		DEVICE (ma ha bisogno di essere inizializzato)
		numNextLevelNodes	int*				DEVICE (ma ha bisogno di essere inizializzato)
		numCurrLevelNodes	int					HOST -> parametro della funzione al device
	*/


	CudaContext(CSRGraph &G){
		nodeCount = G.nodeCount;


		CHECK_CUDA_ERROR(cudaMalloc((void**)&nodePtrs,       sizeof(int) * (G.nodeCount+1)));
		CHECK_CUDA_ERROR(cudaMalloc((void**)&nodeNeighbors,  sizeof(int) * G.edgeCount));
		CHECK_CUDA_ERROR(cudaMalloc((void**)&nodeVisited,    sizeof(int) * G.nodeCount));
		CHECK_CUDA_ERROR(cudaMalloc((void**)&currLevelNodes, sizeof(int) * G.nodeCount));
		CHECK_CUDA_ERROR(cudaMalloc((void**)&nextLevelNodes, sizeof(int) * G.nodeCount));
		CHECK_CUDA_ERROR(cudaMalloc((void**)&numNextLevelNodes, sizeof(int)));


		CHECK_CUDA_ERROR(cudaMemcpy(nodePtrs,           G.nodePtrs, sizeof(int) * (G.nodeCount+1), cudaMemcpyHostToDevice));
		CHECK_CUDA_ERROR(cudaMemcpy(nodeNeighbors, G.nodeNeighbors, sizeof(int) * G.edgeCount,     cudaMemcpyHostToDevice));
	}


	~CudaContext(){
		CHECK_CUDA_ERROR(cudaFree(nodePtrs));
		CHECK_CUDA_ERROR(cudaFree(nodeNeighbors));
		CHECK_CUDA_ERROR(cudaFree(nodeVisited));
		CHECK_CUDA_ERROR(cudaFree(currLevelNodes));
		CHECK_CUDA_ERROR(cudaFree(nextLevelNodes));
		CHECK_CUDA_ERROR(cudaFree(numNextLevelNodes));
	}
};


std::vector<int> cpuReachability(CudaContext &ctx, size_t &kernelTimeMilliseconds);
std::vector<int> gpuReachability(CudaContext &ctx, size_t &kernelTimeMilliseconds);
std::vector<int> gpuReachabilityOptimized(CudaContext &ctx, size_t &kernelTimeMilliseconds);


#endif
