#include <pcap.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <signal.h>
#include <time.h>
#include <sys/time.h>
#include <fcntl.h>
#include <termios.h>

#define WIFI 0
#define ZIG 1
#define PKTS 4444
#define TI_THPT 0xA2
#define TI_SEND 0xA3

unsigned long ms_timer(u_char cmd) {
	static struct timespec times[2];
	unsigned long time_ms = 0;
	clock_gettime(CLOCK_MONOTONIC_RAW, times + cmd);
	if(cmd) time_ms = (times[1].tv_sec - times[0].tv_sec) * 1000
					+ (times[1].tv_nsec - times[0].tv_nsec) / 1000000;
	return time_ms;
}

int main() {
    FILE* fp; pcap_t *wifi_pcap;
    char pcap_error[PCAP_ERRBUF_SIZE];
    u_char wifi_packet[64] = {0}, wifi_local_mac[6], uart_buffer[3];
    int cc2650_uart, i;
    unsigned long thruput[2];

    if ((fp = fopen("/sys/class/net/wlan0/address", "r"))) {
		fscanf(fp, "%02hhx", wifi_local_mac);
		for (i=1; i<6; ++i) fscanf(fp, ":%02hhx", wifi_local_mac + i);
		fclose(fp);
	} else {
		fprintf(stderr, "\nUnable to read local MAC address\n"); exit(1);
	}
    if (NULL == (wifi_pcap = pcap_open_live("wlan0", 1024, 1, 250, pcap_error))) {
		fprintf(stderr, "\nUnable to open adapter: %s\n", pcap_error); exit(1);
	}
    if (-1 == (cc2650_uart = open("/dev/ttyUSB1", O_RDWR | O_NOCTTY | O_NDELAY))) {
		fprintf(stderr, "\nUnable to open /dev/ttyUSB1\n"); exit(1);
	} fcntl(cc2650_uart, F_SETFL, 0);

    uart_buffer[0] = TI_SEND; uart_buffer[1] = 2; uart_buffer[2] = 0;
    write(cc2650_uart, (void *)uart_buffer, sizeof(uart_buffer));

    memset(wifi_packet, 0xFF, 6);
    memcpy(wifi_packet + 6, wifi_local_mac, 6);
    ms_timer(0);
    for(i=0; i<PKTS; ++i) pcap_inject(wifi_pcap, (void *)wifi_packet, sizeof(wifi_packet));
    thruput[WIFI] = PKTS * 1000 / ms_timer(1);
    printf("WIFI: %lu\t", thruput[WIFI]);

    uart_buffer[0] = TI_THPT; uart_buffer[1] = 0; uart_buffer[2] = 0;
    write(cc2650_uart, (void *)uart_buffer, sizeof(uart_buffer));
    read(cc2650_uart, (void *)uart_buffer, sizeof(uart_buffer));
    thruput[ZIG] = uart_buffer[1] << 8;
    thruput[ZIG] |= uart_buffer[2];
    printf("ZIGBEE: %lu\n", thruput[ZIG]);

    fp = fopen("thruput.dat", "w");
    fprintf(fp, "%lu %lu", thruput[WIFI], thruput[ZIG]);
    fclose(fp);
    return 0;
}