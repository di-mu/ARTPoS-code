all:
	gcc wz.c main.c -o wz -lpcap -Wall
	gcc thruput.c -o thruput -lpcap -Wall