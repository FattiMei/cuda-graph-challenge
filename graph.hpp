#ifndef __GRAPH_H_INCLUDED__
#define __GRAPH_H_INCLUDED__


#include <iostream>
#include <set>
#include <vector>


typedef std::pair<int,int> Edge;


struct SimpleGraph{
	int nodeCount;
	int edgeCount;
	std::set<Edge> Edges;


	void printNeighborCount(){
		std::vector<int> nodeNeighborCount(nodeCount,0);


		for(Edge e : Edges){
			++nodeNeighborCount[e.first];
		}


		for(int i = 0; i < nodeCount; ++i){
			std::cout
				<< "il nodo "
				<< i
				<< " ha "
				<< nodeNeighborCount[i]
				<< " vicini"
				<< std::endl;
		}
	}
};


struct CSRGraph{
	int nodeCount;
	int edgeCount;
	int *nodePtrs;
	int *nodeNeighbors;


	CSRGraph(SimpleGraph &G){
		nodeCount     = G.nodeCount;
		edgeCount     = G.edgeCount;
		nodePtrs      = new int[G.nodeCount+1];
		nodeNeighbors = new int[G.edgeCount];


		int currentNode = 0;
		int previousNode = -1;
		int i = 0;


		// itero ordinatamente nel set
		for(Edge e : G.Edges){
			currentNode = e.first;


			if(currentNode != previousNode){
				for(int u = previousNode + 1; u <= currentNode; ++u){
					nodePtrs[u] = i;
				}

				previousNode = currentNode;
			}


			nodeNeighbors[i] = e.second;
			++i;
		}


		// deal with remaining nodes
		for(; currentNode < G.nodeCount; ++currentNode){
			nodePtrs[currentNode+1] = i;
		}
	}


	void printNeighborCount(){
		for(int i = 0; i < nodeCount; ++i){
			std::cout
				<< "il nodo "
				<< i
				<< " ha "
				<< nodePtrs[i+1] - nodePtrs[i]
				<< " vicini"
				<< std::endl;
		}
	}


	~CSRGraph(){
		delete[] nodePtrs;
		delete[] nodeNeighbors;
	}
};


#endif
