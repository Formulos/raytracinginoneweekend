OBJS	= main.o
OBJSp	= lol.o
SOURCE	= main.cc
OUT	= out
CC	 = g++
FLAGS	 = -g -c -Wall -fopenmp
LFLAGS	 = 

all: $(OBJS)
	$(CC) -g -o3 -fopenmp $(OBJS) -o $(OUT) $(LFLAGS)

main.o: main.cc
	$(CC) $(FLAGS) main.cc

npara: main.cc
	$(CC) -g -c -Wall main.cc -o np.o
	$(CC) -g np.o -o n_out
	rm -f np.o

clean:
	rm -f $(OBJS) $(OUT)
	rm -f np.o
	rm -f n_out.o
