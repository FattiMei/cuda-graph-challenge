#include "reachability.hpp"
#include "error.hpp"


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
				++(*numNextLevelNodes);


				// è garantito che l'elemento non sia duplicato nella coda perchè è segnato come visitato
			}
		}
	}
}


std::vector<int> cpuReachability(CudaGraph &G){
	std::vector<int> result(G.nodeCount, 0);


	int *currLevelNodes = new int[G.nodeCount];
	int *nextLevelNodes = new int[G.nodeCount];
	int *nodeVisited    = result.data();
	int numCurrLevelNodes;
	int numNextLevelNodes;


	// inizializzazione della coda
	numCurrLevelNodes = 1;
	currLevelNodes[0] = 0;


	while(numCurrLevelNodes != 0){
		numNextLevelNodes = 0;

		cpuKernel(
				G.nodePtrs
				,G.nodeNeighbors
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


	return result;
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


std::vector<int> gpuReachability(CudaGraph &G){
	std::vector<int> result(G.nodeCount, 0);


	/*
		Nome				Tipo				Relazione
		nodePtrs			int[nodeCount+1]	HOST -> DEVICE
		nodeNeighbors		int[nodeCount]		HOST -> DEVICE
		nodeVisited			int[nodeCount]		DEVICE -> HOST (ma ha bisogno di essere inizializzato)
		currLevelNodes		int[nodeCount]		DEVICE
		nextLevelNodes		int[nodeCount]		DEVICE (ma ha bisogno di essere inizializzato)
		numNextLevelNodes	int*				DEVICE (ma ha bisogno di essere inizializzato)
		numCurrLevelNodes	int					HOST -> parametro della funzione al device
	*/


	int *nodePtrs = NULL;
	int *nodeNeighbors = NULL;
	int *nodeVisited = NULL;
	int *currLevelNodes = NULL;
	int *nextLevelNodes = NULL;
	int *numNextLevelNodes = NULL;
	int numCurrLevelNodes;


	CHECK_CUDA_ERROR(cudaMalloc(&nodePtrs,       sizeof(int) * (G.nodeCount+1)));
	CHECK_CUDA_ERROR(cudaMalloc(&nodeNeighbors,  sizeof(int) * G.nodeCount));
	CHECK_CUDA_ERROR(cudaMalloc(&nodeVisited,    sizeof(int) * G.nodeCount));
	CHECK_CUDA_ERROR(cudaMalloc(&currLevelNodes, sizeof(int) * G.nodeCount));
	CHECK_CUDA_ERROR(cudaMalloc(&nextLevelNodes, sizeof(int) * G.nodeCount));
	CHECK_CUDA_ERROR(cudaMalloc(&numNextLevelNodes, sizeof(int)));


	CHECK_CUDA_ERROR(cudaMemcpy(nodePtrs,           G.nodePtrs, sizeof(int) * (G.nodeCount+1), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(nodeNeighbors, G.nodeNeighbors, sizeof(int) * G.nodeCount, cudaMemcpyHostToDevice));


	// inizializzazione della coda
	numCurrLevelNodes = 1;


	// currLevelNodes[0] = 0;
	CHECK_CUDA_ERROR(cudaMemset(currLevelNodes, 0, sizeof(int)));
	CHECK_CUDA_ERROR(cudaMemset(nodeVisited, 0, sizeof(int) * G.nodeCount));


	while(numCurrLevelNodes != 0){
		// numNextLevelNodes = 0;
		CHECK_CUDA_ERROR(cudaMemset(numNextLevelNodes, 0, sizeof(int)));

		gpuKernel<<<1, 1024>>>(
				 nodePtrs
				,nodeNeighbors
				,nodeVisited
				,currLevelNodes
				,nextLevelNodes
				,numCurrLevelNodes
				,numNextLevelNodes
				);

		// numCurrLevelNodes = *numNextLevelNodes;
		CHECK_CUDA_ERROR(cudaMemcpy(&numCurrLevelNodes, numNextLevelNodes, sizeof(int), cudaMemcpyDeviceToHost));

		std::swap(currLevelNodes, nextLevelNodes);
	}


	CHECK_CUDA_ERROR(cudaFree(nodePtrs));
	CHECK_CUDA_ERROR(cudaFree(nodeNeighbors));
	CHECK_CUDA_ERROR(cudaFree(nodeVisited));
	CHECK_CUDA_ERROR(cudaFree(currLevelNodes));
	CHECK_CUDA_ERROR(cudaFree(nextLevelNodes));
	CHECK_CUDA_ERROR(cudaFree(numNextLevelNodes));


	return result;
}
