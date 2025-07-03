CC = gcc
FLAGS = -std=c17 -Iinclude -Wall

all: main.o activation.o loss.o matrix.o nn.o
	$(CC) $(FLAGS) main.o activation.o loss.o matrix.o nn.o -o main -lm

main.o: example/main.c include/CNet.h
	$(CC) $(FLAGS) -c example/main.c -o main.o

activation.o: src/activation.c include/CNet.h
	$(CC) $(FLAGS) -c src/activation.c -o activation.o

loss.o: src/loss.c include/CNet.h
	$(CC) $(FLAGS) -c src/loss.c -o loss.o

matrix.o: src/matrix.c include/CNet.h
	$(CC) $(FLAGS) -c src/matrix.c -o matrix.o

nn.o: src/nn.c include/CNet.h
	$(CC) $(FLAGS) -c src/nn.c -o nn.o

clean:
	rm -f main main.o activation.o loss.o matrix.o nn.o