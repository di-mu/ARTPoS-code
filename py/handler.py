import sys, random, time
from restructure import search
from online_opt_mod import search_test

row = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
col = [-9, -6, -3, 0, 1, 2, 3, 4, 5]

if len(sys.argv) != 2:
	print("python3 -W ignore handler.py <1: ART-WiFi, 2: ART-ZigBee, 3: ART-WZ, 4: Fixed Power-WZ 5: ARTPoS 6: ARTPoS-irp 7: benchmark(10000x)>")
	quit()
f = open("c.dat", "r")
data = f.readline().split()
f.close()
f = open("py.dat", "w")
cmd = int(sys.argv[1])
if cmd == 7:
	for i in range(0,10000):
		start = time.time()
		search(row[random.randint(0,21)], col[random.randint(0,8)], 
			float(random.randint(0,100)) / 100, float(random.randint(0,100)) / 100, 
			random.randint(100,900), random.randint(750,950), random.randint(200,250))
		f.write(str(time.time() - start) + ",")
	quit()
if cmd == 5 || cmd == 6:
	if cmd == 5:
		search_func = search_test
	else
		search_func = search
	(x, y) = search_func(row[int(data[0]) + 1], col[int(data[1]) + 1], 
		float(data[2]) / 100, float(data[3]) / 100, 
		int(data[4]), int(data[5]), int(data[6]))
	f.write(str(row.index(x) - 1) + " " + str(col.index(y) - 1))
	f.close()
	quit()
if cmd == 4:
	f.write("20 7")
	f.close()
	quit()
highest = [20, 7]
for idx in range(0, 2):
	if cmd==3 or cmd==idx+1:
		power = int(data[idx])
		prr = int(data[idx+2])
		step = 2 - idx
		if (prr < 90) and (power <= highest[idx] - step):
			power += step
		if (prr > 95) and (power >= step):
			power -= step
		f.write(str(power))
	else:
		f.write("-1")
	if idx==0:
		f.write(" ")
f.close()
