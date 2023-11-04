all:
	nvcc -DUNDIRECTED -DONE_INDEXED   -o test_undirected_one_indexed main.cpp reachability.cu parse.cpp -O2
	nvcc              -DONE_INDEXED   -o test_directed_one_indexed   main.cpp reachability.cu parse.cpp -O2
	nvcc -DUNDIRECTED -DZERO_INDEXED  -o test_undirected_zero_indexed main.cpp reachability.cu parse.cpp -O2
	nvcc              -DZERO_INDEXED  -o test_directed_zero_indexed   main.cpp reachability.cu parse.cpp -O2


debug:
	g++ -DUNDIRECTED -DONE_INDEXED  -fsanitize=address -o test_undirected_one_indexed main.cpp reachability.cu parse.cpp -O2
	g++              -DONE_INDEXED  -fsanitize=address -o test_directed_one_indexed   main.cpp reachability.cu parse.cpp -O2
	g++ -DUNDIRECTED -DZERO_INDEXED  -fsanitize=address -o test_undirected_zero_indexed main.cpp reachability.cu parse.cpp -O2
	g++              -DZERO_INDEXED  -fsanitize=address -o test_directed_zero_indexed   main.cpp reachability.cu parse.cpp -O2


run: all
	./test_undirected_one_indexed < datasets/ca-AstroPh.mtx
	./test_undirected_one_indexed < datasets/ca-CondMat.mtx
	./test_undirected_one_indexed < datasets/ca-GrQc.mtx
	./test_undirected_one_indexed < datasets/ca-HepTh.mtx
	./test_undirected_one_indexed < datasets/email-Enron.mtx
	./test_undirected_one_indexed < datasets/roadNet-CA.mtx
	./test_undirected_one_indexed < datasets/roadNet-PA.mtx
	./test_undirected_one_indexed < datasets/roadNet-TX.mtx
	./test_directed_one_indexed   < datasets/web-Google.mtx
	./test_directed_one_indexed   < datasets/web-Stanford.mtx
	./test_directed_one_indexed   < datasets/soc-Epinions1.mtx
	./test_undirected_zero_indexed < datasets/standard4.txt
	./test_undirected_zero_indexed < datasets/standard5.txt
	./test_undirected_zero_indexed < datasets/standard6.txt
	./test_undirected_zero_indexed < datasets/out.txt
	./test_undirected_zero_indexed < datasets/erdos_graph.txt
