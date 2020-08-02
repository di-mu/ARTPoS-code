#!/bin/bash

file="dev_id.cnf"
while read -r line
do dev_id="$line"
done < "$file"

# --- WiFi Setup ---
ifconfig wlan0 down
iwconfig wlan0 mode Ad-hoc essid ad_hoc channel 6
ifconfig wlan0 inet 192.168.22.$dev_id/24 up
iwconfig wlan0 txpower 20

# --- Bluetooth Stop ---
service bluetooth stop

# --- CC2650 Connect ---
modprobe ftdi_sio vendor=0x0403 product=0xa6d1
chmod 777 /sys/bus/usb-serial/drivers/ftdi_sio/new_id
echo 0403 a6d1 > /sys/bus/usb-serial/drivers/ftdi_sio/new_id
sleep 1s
chmod 666 /dev/ttyUSB1
screen /dev/ttyUSB1 115200
