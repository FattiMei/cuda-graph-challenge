#include <iostream>
#include <string>
#include <set>
#include <random>


typedef std::pair<unsigned int, unsigned int> Edge;


const char *usage = "Usage: ./generator nodeCount edgeCount";


bool parseCommandLine(int argc, char *argv[], unsigned int &nodeCount, unsigned int &edgeCount){
	if(argc != 3){
		std::cerr << usage << std::endl;
		return false;
	}


	nodeCount = std::stoul(argv[1]);
	edgeCount = std::stoul(argv[2]);


	if(edgeCount > nodeCount * nodeCount){
		std::cerr
			<< "impossible to produce a graph with that many edges"
			<< std::endl;

		return false;
	}


	return true;
}


int main(int argc, char *argv[]){
	unsigned int nodeCount;
	unsigned int edgeCount;


	if(not parseCommandLine(argc, argv, nodeCount, edgeCount)){
		return 1;
	}


	// uniform RNG, maybe in the future we will need a seed
	std::default_random_engine generator;
	std::uniform_int_distribution<unsigned int> U(0,nodeCount);


	std::set<Edge> extractedEdges;


	unsigned int i = 0;
	unsigned int rolls = 0;

	while(i < edgeCount){
		unsigned int u = U(generator);
		unsigned int v = U(generator);

		if(u != v){
			Edge e = {u,v};

			if(extractedEdges.count(e) == 0){
				extractedEdges.insert(e);
				++i;
			}
		}

		++rolls;
	}


	// debug info
	std::cerr
		<< "% rolls for "
		<< edgeCount
		<< " edges between "
		<< nodeCount
		<< " nodes: "
		<< rolls
		<< std::endl;
	

	// print the graph in standard format
	std::cout 
		<< nodeCount 
		<< " "
		<< nodeCount 
		<< " "
		<< edgeCount 
		<< std::endl;


	for(Edge e : extractedEdges){
		std::cout
			<< e.first
			<< " "
			<< e.second
			<< std::endl;
	}


	return 0;
}
