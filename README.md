# ARTPoS

ARTPoS is the system implementation of the radio selection solution presented in papers
"Adaptive Radio and Transmission Power Selection for Internet of Things" (IWQoS'17) and
"Robust Optimal Selection of Radio Type and Transmission Power for Internet of Things" (TOSN'19).

## Setup on Raspberry Pi

1. Install package "libpcap-dev" on Raspberry Pi;
2. Install python3 and numpy on Raspberry Pi;
3. Run "make" in this directory.

## Setup on CC2650

1. Download Contiki OS system from https://github.com/contiki-os/contiki ;
2. Copy "artpos-cc2650" directory to "contiki/examples/";
3. Run "make TARGET=srf06-cc26xx BOARD=srf06/cc26xx" in "contiki/examples/artpos-cc2650" directory;
4. Program the flash memory of a CC2650 development board with the generated "*.hex" file (from step 3)
using TI's "SmartRF Flash Programmer 2" tool.

## Configurations

1. The number in "dev_id.cnf" is the ID of the current device (from 0).
If the ID is an odd number, the device is an end device (transmitter);
If the ID is an even number, the device is a gateway (receiver).

2. The string in "target_mac.cnf" is the WiFi MAC address of the gateway
or "ff:ff:ff:ff:ff:ff" to use the broadcasting mode for WiFi transmissions.

## Execution on two devices

1. On the gateway:

```bash
# ./wz-setup.sh
# ./wz
```

2. On the end device:

```bash
# ./wz-setup.sh
# ./wz <mode> <number of packets> <number of batches>
```

The numbers for "mode":
1:ART-WiFi, 2:ART-ZigBee, 3:ART-WiFi-ZigBee 4:Fixed-Power-WiFi-ZigBee 5:ARTPoS 6:ARTPoS-irp 7:micro-benchmark experiment

The following example uses ARTPoS for radio selection and transmits 3000 packets for 30 batches:

```bash
# ./wz 5 3000 30
```
