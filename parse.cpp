#include "graph.hpp"
#include "parse.hpp"
#include <limits>


void ignoreComments(std::istream &in){
	bool exit = false;


	while(exit == false){
		char c = in.peek();


		if(c == '%'){
			in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		}
		else{
			exit = true;
		}
	}
}


void readHeader(std::istream &in, int &nodeCount, int &edgeCount){
	unsigned int rowCount;
	unsigned int colCount;


	in >> rowCount >> colCount >> edgeCount;


	if(rowCount == colCount){
		nodeCount = rowCount;
	}
	else{
		std::cerr << "number of rows must be equal to number of columns" << std::endl;
		exit(EXIT_FAILURE);
	}
}


void readEdges(std::istream &in, int edgeCountInFile, std::set<Edge> &E){
	E.clear();


	for(int i = 0; i < edgeCountInFile; ++i){
		unsigned int u;
		unsigned int v;


		in >> u >> v;


#ifdef ZERO_INDEXED
			--u; --v;
#endif


		E.insert({u,v});


#ifdef UNDIRECTED
		E.insert({v,u});
#endif
	}
}


SimpleGraph parseGraph(std::istream &in){
	SimpleGraph result;
	int edgeCountInFile;


	ignoreComments(in);
	readHeader    (in, result.nodeCount, edgeCountInFile);
	readEdges     (in, edgeCountInFile, result.Edges);


	result.edgeCount = result.Edges.size();


	return result;
}
