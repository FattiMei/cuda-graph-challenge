all:
	g++ -DUNDIRECTED -DONE_INDEXED  -fsanitize=address -o test_undirected_one_indexed main.cpp reachability.cpp parse.cpp -O2
	g++              -DONE_INDEXED  -fsanitize=address -o test_directed_one_indexed   main.cpp reachability.cpp parse.cpp -O2


run: all
	./test_undirected_one_indexed < datasets/email-Enron.mtx
	./test_undirected_one_indexed < datasets/roadNet-CA.mtx
	./test_undirected_one_indexed < datasets/roadNet-PA.mtx
	./test_undirected_one_indexed < datasets/roadNet-TX.mtx
	./test_directed_one_indexed   < datasets/web-Google.mtx
	./test_directed_one_indexed   < datasets/web-Stanford.mtx
