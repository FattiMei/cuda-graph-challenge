#include "reachability.hpp"


void cpuKernel(
	 int *nodePtrs
	,int *nodeNeighbors
	,int *nodeVisited
	,int *currLevelNodes
	,int *nextLevelNodes
	,const unsigned int numCurrLevelNodes
	,int *numNextLevelNodes){


	for(int i = 0; i < numCurrLevelNodes; ++i){
		const int u = currLevelNodes[i];
		const int neighborCount = nodePtrs[u+1] - nodePtrs[u];


		for(int j = 0; j < neighborCount; ++j){
			const int v = nodeNeighbors[nodePtrs[u] + j];

			if(nodeVisited[v] == 0){
				nodeVisited[v] = 1;


				// aggiunta alla coda
				nextLevelNodes[*numNextLevelNodes] = v;
				++(numNextLevelNodes);


				// è garantito che l'elemento non sia duplicato nella coda perchè è segnato come visitato
			}
		}
	}
}


void cpuReachability(CudaGraph G, std::vector<int> &nodeVisited){
}


__global__ void gpuKernel(
	 int *nodePtrs
	,int *nodeNeighbors
	,int *nodeVisited
	,int *currLevelNodes
	,int *nextLevelNodes
	,const unsigned int numCurrLevelNodes
	,int *numNextLevelNodes){


	// solo blocchi lineari
	int i = threadIdx.x + blockIdx.x * blockDim.x;


	if(i < numCurrLevelNodes){
		const int u = currLevelNodes[i];
		const int neighborCount = nodePtrs[u+1] - nodePtrs[u];


		for(int j = 0; i < neighborCount; ++j){
			const int v = nodeNeighbors[nodePtrs[u] + j];


			if(atomicCAS(nodeVisited + v, 0, 1) == 0){
				// sono il primo che fa la modifica, lo metto in coda
				int queuePtr = atomicAdd(numNextLevelNodes, 1);


				// sono l'unico che scrive qua dentro
				nextLevelNodes[queuePtr] = v;
			}
		}
	}
}
