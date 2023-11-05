#include "reachability.hpp"
#include "error.hpp"
#include <chrono>


using namespace std::chrono;


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


std::vector<int> cpuReachability(CSRGraph &G, size_t &kernelTimeMilliseconds){
	std::vector<int> result(G.nodeCount, 0);
	std::chrono::high_resolution_clock clock;


	int *currLevelNodes = new int[G.nodeCount];
	int *nextLevelNodes = new int[G.nodeCount];
	int *nodeVisited    = result.data();
	int numCurrLevelNodes;
	int numNextLevelNodes;


	// INIZIO MISURA
	const auto t0 = clock.now();

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


	// FINE MISURA
	const auto t1 = clock.now();
	kernelTimeMilliseconds = duration_cast<microseconds>(t1-t0).count();


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


		// occhio al contatore di questo ciclo, era qui il bug
		for(int j = 0; j < neighborCount; ++j){
			const int v = nodeNeighbors[nodePtrs[u] + j];


			if(atomicCAS(&nodeVisited[v], 0, 1) == 0){
				// sono il primo che fa la modifica, lo metto in coda
				int queuePtr = atomicAdd(numNextLevelNodes, 1);


				// sono l'unico che scrive qua dentro
				nextLevelNodes[queuePtr] = v;
			}
		}
	}
}


__global__ void gpuKernelOptimizedIteration(
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
		int *begin = nodeNeighbors + nodePtrs[u];
		int *end   = nodeNeighbors + nodePtrs[u+1];


		for(int *v = begin; v < end; ++v){
			if(atomicCAS(nodeVisited + (*v), 0, 1) == 0){
				// sono il primo che fa la modifica, lo metto in coda
				int queuePtr = atomicAdd(numNextLevelNodes, 1);


				// sono l'unico che scrive qua dentro
				nextLevelNodes[queuePtr] = *v;
			}
		}
	}
}


__global__ void gpuKernelOptimizedShared(
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
		int *begin = nodeNeighbors + nodePtrs[u];
		int *end   = nodeNeighbors + nodePtrs[u+1];


		for(int *v = begin; v < end; ++v){
			if(atomicCAS(nodeVisited + (*v), 0, 1) == 0){
				// sono il primo che fa la modifica, lo metto in coda
				int queuePtr = atomicAdd(numNextLevelNodes, 1);


				// sono l'unico che scrive qua dentro
				nextLevelNodes[queuePtr] = *v;
			}
		}
	}
}


std::vector<int> gpuReachability(CudaContext &ctx, size_t &kernelTimeMilliseconds){
	std::vector<int> result(ctx.nodeCount, 0);
	std::chrono::high_resolution_clock clock;


	// INIZIO MISURA
	const auto t0 = clock.now();

	// inizializzazione della coda
	ctx.numCurrLevelNodes = 1;


	// currLevelNodes[0] = 0;
	CHECK_CUDA_ERROR(cudaMemset(ctx.currLevelNodes, 0, sizeof(int)));
	CHECK_CUDA_ERROR(cudaMemset(ctx.nodeVisited, 0, sizeof(int) * ctx.nodeCount));


	while(ctx.numCurrLevelNodes != 0){
		// numNextLevelNodes = 0;
		CHECK_CUDA_ERROR(cudaMemset(ctx.numNextLevelNodes, 0, sizeof(int)));

		int threadsPerBlock = 64;
		int blockSize = (ctx.numCurrLevelNodes + threadsPerBlock - 1) / threadsPerBlock;

		gpuKernel<<<blockSize, threadsPerBlock>>>(
				ctx.nodePtrs
				,ctx.nodeNeighbors
				,ctx.nodeVisited
				,ctx.currLevelNodes
				,ctx.nextLevelNodes
				,ctx.numCurrLevelNodes
				,ctx.numNextLevelNodes
		);

		CHECK_CUDA_ERROR(cudaPeekAtLastError());


		// numCurrLevelNodes = *numNextLevelNodes;
		CHECK_CUDA_ERROR(cudaMemcpy(&ctx.numCurrLevelNodes, ctx.numNextLevelNodes, sizeof(int), cudaMemcpyDeviceToHost));


		std::swap(ctx.currLevelNodes, ctx.nextLevelNodes);
	}


	// FINE MISURA
	const auto t1 = clock.now();
	kernelTimeMilliseconds = duration_cast<microseconds>(t1-t0).count();


	CHECK_CUDA_ERROR(cudaMemcpy(result.data(), ctx.nodeVisited, ctx.nodeCount * sizeof(int), cudaMemcpyDeviceToHost));



	return result;
}


std::vector<int> gpuReachabilityOpt(CudaContext &ctx, size_t &kernelTimeMilliseconds){
	std::vector<int> result(ctx.nodeCount, 0);
	std::chrono::high_resolution_clock clock;


	// INIZIO MISURA
	const auto t0 = clock.now();

	// inizializzazione della coda
	ctx.numCurrLevelNodes = 1;


	// currLevelNodes[0] = 0;
	CHECK_CUDA_ERROR(cudaMemset(ctx.currLevelNodes, 0, sizeof(int)));
	CHECK_CUDA_ERROR(cudaMemset(ctx.nodeVisited, 0, sizeof(int) * ctx.nodeCount));


	while(ctx.numCurrLevelNodes != 0){
		// numNextLevelNodes = 0;
		CHECK_CUDA_ERROR(cudaMemset(ctx.numNextLevelNodes, 0, sizeof(int)));

		int threadsPerBlock = 64;
		int blockSize = (ctx.numCurrLevelNodes + threadsPerBlock - 1) / threadsPerBlock;

		gpuKernelOptimizedIteration<<<blockSize, threadsPerBlock>>>(
				ctx.nodePtrs
				,ctx.nodeNeighbors
				,ctx.nodeVisited
				,ctx.currLevelNodes
				,ctx.nextLevelNodes
				,ctx.numCurrLevelNodes
				,ctx.numNextLevelNodes
		);

		CHECK_CUDA_ERROR(cudaPeekAtLastError());


		// numCurrLevelNodes = *numNextLevelNodes;
		CHECK_CUDA_ERROR(cudaMemcpy(&ctx.numCurrLevelNodes, ctx.numNextLevelNodes, sizeof(int), cudaMemcpyDeviceToHost));


		std::swap(ctx.currLevelNodes, ctx.nextLevelNodes);
	}


	// FINE MISURA
	const auto t1 = clock.now();
	kernelTimeMilliseconds = duration_cast<microseconds>(t1-t0).count();


	CHECK_CUDA_ERROR(cudaMemcpy(result.data(), ctx.nodeVisited, ctx.nodeCount * sizeof(int), cudaMemcpyDeviceToHost));



	return result;
}


std::vector<int> gpuReachabilityShr(CudaContext &ctx, size_t &kernelTimeMilliseconds){
	std::vector<int> result(ctx.nodeCount, 0);
	std::chrono::high_resolution_clock clock;


	// INIZIO MISURA
	const auto t0 = clock.now();

	// inizializzazione della coda
	ctx.numCurrLevelNodes = 1;


	// currLevelNodes[0] = 0;
	CHECK_CUDA_ERROR(cudaMemset(ctx.currLevelNodes, 0, sizeof(int)));
	CHECK_CUDA_ERROR(cudaMemset(ctx.nodeVisited, 0, sizeof(int) * ctx.nodeCount));


	while(ctx.numCurrLevelNodes != 0){
		// numNextLevelNodes = 0;
		CHECK_CUDA_ERROR(cudaMemset(ctx.numNextLevelNodes, 0, sizeof(int)));

		int threadsPerBlock = 64;
		int blockSize = (ctx.numCurrLevelNodes + threadsPerBlock - 1) / threadsPerBlock;

		gpuKernelOptimizedShared<<<blockSize, threadsPerBlock>>>(
				ctx.nodePtrs
				,ctx.nodeNeighbors
				,ctx.nodeVisited
				,ctx.currLevelNodes
				,ctx.nextLevelNodes
				,ctx.numCurrLevelNodes
				,ctx.numNextLevelNodes
		);

		CHECK_CUDA_ERROR(cudaPeekAtLastError());


		// numCurrLevelNodes = *numNextLevelNodes;
		CHECK_CUDA_ERROR(cudaMemcpy(&ctx.numCurrLevelNodes, ctx.numNextLevelNodes, sizeof(int), cudaMemcpyDeviceToHost));


		std::swap(ctx.currLevelNodes, ctx.nextLevelNodes);
	}


	// FINE MISURA
	const auto t1 = clock.now();
	kernelTimeMilliseconds = duration_cast<microseconds>(t1-t0).count();


	CHECK_CUDA_ERROR(cudaMemcpy(result.data(), ctx.nodeVisited, ctx.nodeCount * sizeof(int), cudaMemcpyDeviceToHost));



	return result;
}
