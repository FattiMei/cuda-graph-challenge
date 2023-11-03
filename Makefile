all:
	nvcc -o test main.cu reachability.cu parse.cpp -O2
	nvcc -DUNDIRECTED -DZERO_INDEXED -o testz main.cu parse.cpp reachability.cu


run: all
	./test < datasets/email-Enron.mtx
	./test < datasets/roadNet-CA.mtx
	./test < datasets/roadNet-PA.mtx
	./test < datasets/roadNet-TX.mtx
	./test < datasets/web-Google.mtx
	./test < datasets/web-Stanford.mtx
