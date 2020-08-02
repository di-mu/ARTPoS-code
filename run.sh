#!/bin/bash
DATAFILE="log.csv"

./wz 4 1000 30 | tee -a "$DATAFILE"
sleep 30s
./wz 5 1000 30 | tee -a "$DATAFILE"
sleep 30s
./wz 4 3000 30 | tee -a "$DATAFILE"
sleep 30s
./wz 5 3000 30 | tee -a "$DATAFILE"
sleep 30s
./wz 4 5000 30 | tee -a "$DATAFILE"
sleep 30s
./wz 5 5000 30 | tee -a "$DATAFILE"
sleep 30s
./wz 4 7000 30 | tee -a "$DATAFILE"
sleep 30s
./wz 5 7000 30 | tee -a "$DATAFILE"
sleep 30s
./wz 4 9000 30 | tee -a "$DATAFILE"
sleep 30s
./wz 5 9000 30 | tee -a "$DATAFILE"
