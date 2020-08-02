#ifndef _WZ_H_
#define _WZ_H_

#include <pcap.h>

#define TIME_BLOCK 10
#define BLOCK_DATA 1
#define BLOCK_REQ 2
#define BLOCK_ACK 3
#define TI_RECV 0xA1
#define TI_SEND 0xA3
#define TI_TXPWR 0xA5
#define TI_THPT 0xA2
#define TI_PCNT 0xA4
#define WIFI 0
#define ZIG 1
#define ACTIVE_DEV (dev_id & 1)

extern long _PKT_DATA, PKT_DATA;
extern int N_BATCH, batch_id;
extern long wifi_split;
extern long thruput[2];
extern long packet_count[2];
extern signed char power_index[2];
extern const int *TXPWR[2];
extern u_char dev_id;

extern int bytes2int(const u_char *, u_char);
extern void int2bytes(int, u_char *, u_char);
extern void wz_init();
extern void wz_end();
extern void wifi_on();
extern void wifi_off();
extern unsigned long ms_timer(u_char);
extern void ms_sleep(unsigned long);
extern void set_txpower(u_char, int);
extern int query_2650(u_char, int);
extern void wz_transmit_block();
extern void wz_receive_block();

#endif