#include "wz.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <termios.h>

#define MAGIC 0x17
#define PORT_UART cc2650_uart, (void *)uart_buffer, sizeof(uart_buffer)
#define PORT_WIFI wifi_pcap, (void *)wifi_packet, sizeof(wifi_packet)
#define IOPUT(PORT,FUN) while(FUN(PORT) <= 0)
#define PCAP_TIMEOUT 200

static pcap_t *wifi_pcap;
static int cc2650_uart;
static struct pcap_pkthdr header;
static u_char wifi_packet[64];
static const u_char *wifi_receive;
static u_char wifi_local_mac[6];
static u_char wifi_target_mac[6];
static u_char uart_buffer[3];
static char syscmd[50];
static const int TXPWR_WIFI[21] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
static const int TXPWR_ZIG[8] = {-6, -3, 0, 1, 2, 3, 4, 5};
const int *TXPWR[2] = {TXPWR_WIFI, TXPWR_ZIG};
signed char power_index[2] = {20, 7};
long thruput[2], packet_count[2] = {0,0};
long _PKT_DATA, PKT_DATA = 5555;
int N_BATCH, batch_id = 0;
long wifi_split;
u_char dev_id = 0;

int bytes2int(const u_char *bytes, u_char size) {
	int result = 0; u_char i;
	for(i=0; i<size; ++i) { result <<= 8; result |= bytes[i]; }
	return result;
}

void int2bytes(int num, u_char *bytes, u_char size) {
	while(size) { bytes[--size] = num & 0xFF; num >>= 8; }
}

void wz_init() {
	FILE *fp; u_char i;
	char pcap_error[PCAP_ERRBUF_SIZE];
	if ((fp = fopen("dev_id.cnf", "r"))) { fscanf(fp, "%hhu", &dev_id); fclose(fp); }
	if ((fp = fopen("/sys/class/net/wlan0/address", "r"))) {
		fscanf(fp, "%02hhx", wifi_local_mac);
		for (i=1; i<6; ++i) fscanf(fp, ":%02hhx", wifi_local_mac + i);
		fclose(fp);
	} else {
		fprintf(stderr, "\nUnable to read local MAC address\n"); exit(1);
	}
	if ((fp = fopen("target_mac.cnf", "r"))) {
		fscanf(fp, "%02hhx", wifi_target_mac);
		for (i=1; i<6; ++i) fscanf(fp, ":%02hhx", wifi_target_mac + i);
		fclose(fp);
	} else {
		fprintf(stderr, "\nUnable to read target MAC address\n"); exit(1);
	}
	if ((fp = fopen("thruput.dat", "r"))) {
		fscanf(fp, "%ld %ld", thruput + WIFI, thruput + ZIG);
		fclose(fp);
	} else {
		fprintf(stderr, "\nUnable to read thruput data\n"); exit(1);
	}
	wifi_on(); set_txpower(WIFI, 21);
	if (NULL == (wifi_pcap = pcap_open_live("wlan0", 1024, 1, PCAP_TIMEOUT, pcap_error))) {
		fprintf(stderr, "\nUnable to open adapter: %s\n", pcap_error); exit(1);
	}
	if (ACTIVE_DEV) wifi_off();
	if (-1 == (cc2650_uart = open("/dev/ttyUSB1", O_RDWR | O_NOCTTY | O_NDELAY))) {
		fprintf(stderr, "\nUnable to open /dev/ttyUSB1\n"); exit(1);
	} fcntl(cc2650_uart, F_SETFL, 0); //O_NONBLOCK
	memset(wifi_packet, 0xFF, 6);
	memcpy(wifi_packet + 6, wifi_local_mac, 6);
	wifi_packet[20] = MAGIC;
	if (ACTIVE_DEV) printf("BATCH\tTIME_MS\t| PWR_W\tPWR_Z\t| RX_W\tTX_W\t| RX_Z\tTX_Z\n");
	else printf("BATCH\t| RX_W\t| RX_Z\n");
}

void wz_end() {
	pcap_close(wifi_pcap);
	close(cc2650_uart);
}

void wifi_on() {
	system("sudo ifconfig wlan0 up"); ms_sleep(100);
}

void wifi_off() {
	ms_sleep(200); system("sudo ifconfig wlan0 down");
}

unsigned long ms_timer(u_char idx) {
	static struct timespec times[8];
	unsigned long time_ms = 0;
	clock_gettime(CLOCK_MONOTONIC_RAW, times + idx);
	if (idx & 1) time_ms = (times[idx].tv_sec - times[idx-1].tv_sec) * 1000
					+ (times[idx].tv_nsec - times[idx-1].tv_nsec) / 1000000;
	return time_ms;
}

void ms_sleep(unsigned long ms) {
	struct timespec t = {ms/1000, (ms%1000)*1000000};
	nanosleep(&t, NULL);
}

void set_txpower(u_char wz, int dbm) {
	if (wz == WIFI) {
		sprintf(syscmd, "sudo iwconfig wlan0 txpower %d", dbm);
		system(syscmd);
	} else if (wz == ZIG) {
		query_2650(TI_TXPWR, dbm + 21);
	} ms_sleep(100);
}

int query_2650(u_char datatype, int value) {
	uart_buffer[0] = datatype;
	int2bytes(value, uart_buffer+1, 2);
	IOPUT(PORT_UART, write);
	if (datatype & 1) return 0;
	IOPUT(PORT_UART, read);
	return bytes2int(uart_buffer+1, 2);
}

void wz_transmit_block() {
	int i, t, retries = 0; unsigned long ts = ms_timer(3); ms_timer(4);
	wifi_split = (power_index[WIFI] >= 0 && power_index[ZIG] >= 0) ?
		PKT_DATA * thruput[WIFI] / (thruput[WIFI] + thruput[ZIG]) :
		(power_index[WIFI] >= 0 ? PKT_DATA : 0);
	int2bytes(++batch_id, wifi_packet+21, 2);
	if (power_index[ZIG] >= 0) {
		set_txpower(ZIG, TXPWR[ZIG][power_index[ZIG]]);
		query_2650(TI_SEND, PKT_DATA - wifi_split);
	}
	if (power_index[WIFI] >= 0) {
		wifi_on(); set_txpower(WIFI, TXPWR[WIFI][power_index[WIFI]]);
		wifi_packet[23] = BLOCK_DATA;
		for (i = 0; i < wifi_split; ++i) IOPUT(PORT_WIFI, pcap_inject);
		wifi_off();
	}
	printf("%d\t%lu\t| ", batch_id, ts);
	for (i=0; i<2; ++i)
		if (power_index[i] >= 0 ) printf("%d\t", TXPWR[i][power_index[i]]);
		else printf("off\t");
	fflush(stdout);
	t = TIME_BLOCK - ms_timer(5) / 1000;
	if (t < 0) t = 0;
	sleep(t + 1);
	wifi_on(); set_txpower(WIFI, 21);
	wifi_packet[23] = BLOCK_REQ;
	do {
		IOPUT(PORT_WIFI, pcap_inject);
		ms_timer(0);
		while (!(wifi_receive = pcap_next(wifi_pcap, &header))
				|| wifi_receive[20] != MAGIC || wifi_receive[23] != BLOCK_ACK) {
			if (ms_timer(1) > 444) {
				if (retries < 7) { retries++; break; }
				else {
					wifi_off();
					packet_count[WIFI] = packet_count[ZIG] = -1;
					printf("| no response received\n"); fflush(stdout);
					return;
				}
			}
		}
	} while (!wifi_receive || wifi_receive[20] != MAGIC || wifi_receive[23] != BLOCK_ACK);
	wifi_off();
	packet_count[WIFI] = bytes2int(wifi_receive+24, 2);
	packet_count[ZIG] = bytes2int(wifi_receive+26, 2);
	printf("| %lu\t%lu\t", packet_count[WIFI], wifi_split);
	printf("| %lu\t%lu\n", packet_count[ZIG], PKT_DATA - wifi_split);
	fflush(stdout); sleep(1);
}
/*
wifi_packet:
[20] MAGIC
[21-22] batch_id
[23] Block Type
[24-25] Packet Count of WIFI (ACK only)
[26-27] Packet Count of ZIG (ACK only)
*/
void wz_receive_block() {
	u_char blocktype;
	int rx_batch_id;
	static int ack_batch_id = -1;
	if (!(wifi_receive = pcap_next(wifi_pcap, &header)) || wifi_receive[20] != MAGIC) return;
	if (batch_id != (rx_batch_id = bytes2int(wifi_receive+21, 2))) {
		batch_id = rx_batch_id;
		packet_count[WIFI] = 0;
	}
	blocktype = wifi_receive[23];
	if (blocktype == BLOCK_DATA) packet_count[WIFI] ++;
	else if (blocktype == BLOCK_REQ) {
		if (ack_batch_id != batch_id) {
			ack_batch_id = batch_id;
			packet_count[ZIG] = query_2650(TI_PCNT, 0);
		}
		wifi_packet[23] = BLOCK_ACK;
		int2bytes(packet_count[WIFI], wifi_packet + 24, 2);
		int2bytes(packet_count[ZIG], wifi_packet+26, 2);
		IOPUT(PORT_WIFI, pcap_inject);
		printf("%d\t| %lu\t| %lu\n", batch_id, packet_count[WIFI], packet_count[ZIG]);
		fflush(stdout);
	}
}
