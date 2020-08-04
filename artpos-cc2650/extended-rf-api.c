/*
 * Modified by Di Mu (dmu1@binghamton.edu) for ARTPoS implementation
 *
 * Copyright (c) 2014, George Oikonomou (george@contiki-os.org)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 */
/**
 *   Example project demonstrating the extended RF API functionality
 */
#include "contiki.h"
#include "net/netstack.h"
#include "dev/radio.h"
#include "sys/clock.h"
#include "dev/cc26xx-uart.h"
#include "button-sensor.h"
//#include "rf-core/rf-ble.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define PKTSIZE 78 // (10 MAC header + 2 batch id + 2 packet id + 64 payload)
#define MAGIC 0x17
#define WAIT 0
#define READY 1
#define ON RADIO_POWER_MODE_ON
#define OFF RADIO_POWER_MODE_OFF
#define N_TASK 10
#define PKT_INRAM 234
#define TI_RECV 0xA1
#define TI_SEND 0xA3
#define TI_TXPWR 0xA5
#define TI_THPT 0xA2
#define TI_PCNT 0xA4
#define TI_RXDAT 0xA6
#define SET_TXPWR(P) set_param(RADIO_PARAM_TXPOWER, P)
#define TURN_RADIO(T) set_param(RADIO_PARAM_POWER_MODE, T)
/*---------------------------------------------------------------------------*/
struct rf_consts {
  radio_value_t channel_min;
  radio_value_t channel_max;
  radio_value_t txpower_min;
  radio_value_t txpower_max;
};

static struct rf_consts consts;
static const struct radio_driver *radio = &ieee_mode_driver;
static unsigned char radio_on = 0;
static unsigned char keep_radio_on = 0;
static unsigned char scmd_state = WAIT;
static unsigned char packet[PKTSIZE] = {0};
static unsigned char uart_buffer[15000]; // holds up to 234 packets
static unsigned long thruput;
static unsigned int packet_count = 0;
static unsigned int batch_id = 0, packet_id = 0;
static radio_value_t value;
static void (*tasks[N_TASK]) () = {NULL};
static uint8_t ext_addr[8];
static const TX_TIMEOUT = 99 * CLOCK_SECOND / 10;
/*---------------------------------------------------------------------------*/
PROCESS(extended_rf_api_process, "Extended RF API demo process");
AUTOSTART_PROCESSES(&extended_rf_api_process);
/*---------------------------------------------------------------------------*/
static void
print_64bit_addr(const uint8_t *addr)
{
  unsigned int i;
  for(i = 0; i < 7; i++) {
    printf("%02x:", addr[i]);
  }
  printf("%02x (network order)\n", addr[7]);
}
/*---------------------------------------------------------------------------*/
static radio_result_t
get_object(radio_param_t param, void *dest, size_t size)
{
  radio_result_t rv;

  rv = NETSTACK_RADIO.get_object(param, dest, size);

  switch(rv) {
  case RADIO_RESULT_ERROR:
    printf("Radio returned an error\n");
    break;
  case RADIO_RESULT_INVALID_VALUE:
    printf("Value is invalid\n");
    break;
  case RADIO_RESULT_NOT_SUPPORTED:
    printf("Param %u not supported\n", param);
    break;
  case RADIO_RESULT_OK:
    break;
  default:
    printf("Unknown return value\n");
    break;
  }

  return rv;
}
/*---------------------------------------------------------------------------*/
static radio_result_t
set_object(radio_param_t param, void *src, size_t size)
{
  radio_result_t rv;

  rv = NETSTACK_RADIO.set_object(param, src, size);

  switch(rv) {
  case RADIO_RESULT_ERROR:
    printf("Radio returned an error\n");
    break;
  case RADIO_RESULT_INVALID_VALUE:
    printf("Value is invalid\n");
    break;
  case RADIO_RESULT_NOT_SUPPORTED:
    printf("Param %u not supported\n", param);
    break;
  case RADIO_RESULT_OK:
    break;
  default:
    printf("Unknown return value\n");
    break;
  }

  return rv;
}
/*---------------------------------------------------------------------------*/
static radio_result_t
get_param(radio_param_t param, radio_value_t *value)
{
  radio_result_t rv;

  rv = NETSTACK_RADIO.get_value(param, value);

  switch(rv) {
  case RADIO_RESULT_ERROR:
    printf("Radio returned an error\n");
    break;
  case RADIO_RESULT_INVALID_VALUE:
    printf("Value %d is invalid\n", *value);
    break;
  case RADIO_RESULT_NOT_SUPPORTED:
    printf("Param %u not supported\n", param);
    break;
  case RADIO_RESULT_OK:
    break;
  default:
    printf("Unknown return value\n");
    break;
  }

  return rv;
}
/*---------------------------------------------------------------------------*/
static radio_result_t
set_param(radio_param_t param, radio_value_t value)
{
  radio_result_t rv;

  rv = NETSTACK_RADIO.set_value(param, value);

  switch(rv) {
  case RADIO_RESULT_ERROR:
    printf("Radio returned an error\n");
    break;
  case RADIO_RESULT_INVALID_VALUE:
    printf("Value %d is invalid\n", value);
    break;
  case RADIO_RESULT_NOT_SUPPORTED:
    printf("Param %u not supported\n", param);
    break;
  case RADIO_RESULT_OK:
    break;
  default:
    printf("Unknown return value\n");
    break;
  }

  return rv;
}
/*---------------------------------------------------------------------------*/
static void
get_rf_consts(void)
{
  printf("====================================\n");
  printf("RF Constants\n");
  printf("Min Channel : ");
  if(get_param(RADIO_CONST_CHANNEL_MIN, &consts.channel_min) == RADIO_RESULT_OK) {
    printf("%3d\n", consts.channel_min);
  }

  printf("Max Channel : ");
  if(get_param(RADIO_CONST_CHANNEL_MAX, &consts.channel_max) == RADIO_RESULT_OK) {
    printf("%3d\n", consts.channel_max);
  }

  printf("Min TX Power: ");
  if(get_param(RADIO_CONST_TXPOWER_MIN, &consts.txpower_min) == RADIO_RESULT_OK) {
    printf("%3d dBm\n", consts.txpower_min);
  }

  printf("Max TX Power: ");
  if(get_param(RADIO_CONST_TXPOWER_MAX, &consts.txpower_max) == RADIO_RESULT_OK) {
    printf("%3d dBm\n", consts.txpower_max);
  }
}
/*---------------------------------------------------------------------------*/
static void
test_off_on(void)
{
  printf("====================================\n");
  printf("Power mode Test: Off, then On\n");

  printf("Power mode is  : ");
  if(get_param(RADIO_PARAM_POWER_MODE, &value) == RADIO_RESULT_OK) {
    if(value == RADIO_POWER_MODE_ON) {
      printf("On\n");
    } else if(value == RADIO_POWER_MODE_OFF) {
      printf("Off\n");
    }
  }

  printf("Turning Off    : ");
  value = RADIO_POWER_MODE_OFF;
  set_param(RADIO_PARAM_POWER_MODE, value);
  if(get_param(RADIO_PARAM_POWER_MODE, &value) == RADIO_RESULT_OK) {
    if(value == RADIO_POWER_MODE_ON) {
      printf("On\n");
    } else if(value == RADIO_POWER_MODE_OFF) {
      printf("Off\n");
    }
  }

  printf("Turning On     : ");
  value = RADIO_POWER_MODE_ON;
  set_param(RADIO_PARAM_POWER_MODE, value);
  if(get_param(RADIO_PARAM_POWER_MODE, &value) == RADIO_RESULT_OK) {
    if(value == RADIO_POWER_MODE_ON) {
      printf("On\n");
    } else if(value == RADIO_POWER_MODE_OFF) {
      printf("Off\n");
    }
  }
}
/*---------------------------------------------------------------------------*/
static void
test_channels(void)
{
  int i;

  printf("====================================\n");
  printf("Channel Test: [%u , %u]\n", consts.channel_min, consts.channel_max);

  for(i = consts.channel_min; i <= consts.channel_min; i++) {
    value = i;
    printf("Switch to: %d, Now: ", value);
    set_param(RADIO_PARAM_CHANNEL, value);
    if(get_param(RADIO_PARAM_CHANNEL, &value) == RADIO_RESULT_OK) {
      printf("%d\n", value);
    }
  }
}
/*---------------------------------------------------------------------------*/
static void
test_rx_modes(void)
{
  int i;

  printf("====================================\n");
  printf("RX Modes Test: [0 , 3]\n");

  for(i = 0; i <= 0; i++) {
    value = i;
    printf("Switch to: %d, Now: ", value);
    set_param(RADIO_PARAM_RX_MODE, value);
    if(get_param(RADIO_PARAM_RX_MODE, &value) == RADIO_RESULT_OK) {
      printf("Address Filtering is ");
      if(value & RADIO_RX_MODE_ADDRESS_FILTER) {
        printf("On, ");
      } else {
        printf("Off, ");
      }
      printf("Auto ACK is ");
      if(value & RADIO_RX_MODE_AUTOACK) {
        printf("On, ");
      } else {
        printf("Off, ");
      }

      printf("(value=%d)\n", value);
    }
  }
}
/*---------------------------------------------------------------------------*/
static void
test_tx_powers(void)
{
  int i;

  printf("====================================\n");
  printf("TX Power Test: [%d , %d]\n", consts.txpower_min, consts.txpower_max);

  for(i = consts.txpower_min; i <= consts.txpower_max; i += 5) {
    value = i;
    printf("Switch to: %3d dBm, Now: ", value);
    set_param(RADIO_PARAM_TXPOWER, value);
    if(get_param(RADIO_PARAM_TXPOWER, &value) == RADIO_RESULT_OK) {
      printf("%3d dBm\n", value);
    }
  }
}
/*---------------------------------------------------------------------------*/
static void
test_cca_thresholds(void)
{
  printf("====================================\n");
  printf("CCA Thres. Test: -105, then -81\n");

  value = -105;
  printf("Switch to: %4d dBm, Now: ", value);
  set_param(RADIO_PARAM_CCA_THRESHOLD, value);
  if(get_param(RADIO_PARAM_CCA_THRESHOLD, &value) == RADIO_RESULT_OK) {
    printf("%4d dBm [0x%04x]\n", value, (uint16_t)value);
  }

  value = -81;
  printf("Switch to: %4d dBm, Now: ", value);
  set_param(RADIO_PARAM_CCA_THRESHOLD, value);
  if(get_param(RADIO_PARAM_CCA_THRESHOLD, &value) == RADIO_RESULT_OK) {
    printf("%4d dBm [0x%04x]\n", value, (uint16_t)value);
  }
}
/*---------------------------------------------------------------------------*/
static void
test_pan_id(void)
{
  radio_value_t new_val;

  printf("====================================\n");
  printf("PAN ID Test: Flip bytes and back\n");

  printf("PAN ID is: ");
  if(get_param(RADIO_PARAM_PAN_ID, &value) == RADIO_RESULT_OK) {
    printf("0x%02x%02x\n", (value >> 8) & 0xFF, value & 0xFF);
  }

  new_val = (value >> 8) & 0xFF;
  new_val |= (value & 0xFF) << 8;
  printf("Switch to: 0x%02x%02x, Now: ", (new_val >> 8) & 0xFF, new_val & 0xFF);
  set_param(RADIO_PARAM_PAN_ID, new_val);
  if(get_param(RADIO_PARAM_PAN_ID, &value) == RADIO_RESULT_OK) {
    printf("0x%02x%02x\n", (value >> 8) & 0xFF, value & 0xFF);
  }

  new_val = (value >> 8) & 0xFF;
  new_val |= (value & 0xFF) << 8;
  printf("Switch to: 0x%02x%02x, Now: ", (new_val >> 8) & 0xFF, new_val & 0xFF);
  set_param(RADIO_PARAM_PAN_ID, new_val);
  if(get_param(RADIO_PARAM_PAN_ID, &value) == RADIO_RESULT_OK) {
    printf("0x%02x%02x\n", (value >> 8) & 0xFF, value & 0xFF);
  }
}
/*---------------------------------------------------------------------------*/
static void
test_16bit_addr(void)
{
  radio_value_t new_val;

  printf("====================================\n");
  printf("16-bit Address Test: Flip bytes and back\n");

  printf("16-bit Address is: ");
  if(get_param(RADIO_PARAM_16BIT_ADDR, &value) == RADIO_RESULT_OK) {
    printf("0x%02x%02x\n", (value >> 8) & 0xFF, value & 0xFF);
  }

  new_val = (value >> 8) & 0xFF;
  new_val |= (value & 0xFF) << 8;
  printf("Switch to: 0x%02x%02x, Now: ", (new_val >> 8) & 0xFF, new_val & 0xFF);
  set_param(RADIO_PARAM_16BIT_ADDR, new_val);
  if(get_param(RADIO_PARAM_16BIT_ADDR, &value) == RADIO_RESULT_OK) {
    printf("0x%02x%02x\n", (value >> 8) & 0xFF, value & 0xFF);
  }

  new_val = (value >> 8) & 0xFF;
  new_val |= (value & 0xFF) << 8;
  printf("Switch to: 0x%02x%02x, Now: ", (new_val >> 8) & 0xFF, new_val & 0xFF);
  set_param(RADIO_PARAM_16BIT_ADDR, new_val);
  if(get_param(RADIO_PARAM_16BIT_ADDR, &value) == RADIO_RESULT_OK) {
    printf("0x%02x%02x\n", (value >> 8) & 0xFF, value & 0xFF);
  }
}
/*---------------------------------------------------------------------------*/
static void
test_64bit_addr(void)
{
  int i;
  uint8_t new_val[8];

  printf("====================================\n");
  printf("64-bit Address Test: Invert byte order\n");

  printf("64-bit Address is: ");
  if(get_object(RADIO_PARAM_64BIT_ADDR, ext_addr, 8) == RADIO_RESULT_OK) {
    print_64bit_addr(ext_addr);
  }

  for(i = 0; i <= 7; i++) {
    new_val[7 - i] = ext_addr[i];
  }

  printf("Setting to       : ");
  print_64bit_addr(new_val);

  printf("64-bit Address is: ");
  set_object(RADIO_PARAM_64BIT_ADDR, new_val, 8);
  if(get_object(RADIO_PARAM_64BIT_ADDR, ext_addr, 8) == RADIO_RESULT_OK) {
    print_64bit_addr(ext_addr);
  }
}
/*---------------------------------------------------------------------------*/
static void
print_rf_values(void)
{
  printf("====================================\n");
  printf("RF Values\n");

  printf("Power: ");
  if(get_param(RADIO_PARAM_POWER_MODE, &value) == RADIO_RESULT_OK) {
    if(value == RADIO_POWER_MODE_ON) {
      printf("On\n");
    } else if(value == RADIO_POWER_MODE_OFF) {
      printf("Off\n");
    }
  }

  printf("Channel: ");
  if(get_param(RADIO_PARAM_CHANNEL, &value) == RADIO_RESULT_OK) {
    printf("%d\n", value);
  }

  printf("PAN ID: ");
  if(get_param(RADIO_PARAM_PAN_ID, &value) == RADIO_RESULT_OK) {
    printf("0x%02x%02x\n", (value >> 8) & 0xFF, value & 0xFF);
  }

  printf("16-bit Address: ");
  if(get_param(RADIO_PARAM_16BIT_ADDR, &value) == RADIO_RESULT_OK) {
    printf("0x%02x%02x\n", (value >> 8) & 0xFF, value & 0xFF);
  }

  printf("64-bit Address: ");
  if(get_object(RADIO_PARAM_64BIT_ADDR, ext_addr, 8) == RADIO_RESULT_OK) {
    print_64bit_addr(ext_addr);
  }

  printf("RX Mode: ");
  if(get_param(RADIO_PARAM_RX_MODE, &value) == RADIO_RESULT_OK) {
    printf("Address Filtering is ");
    if(value & RADIO_RX_MODE_ADDRESS_FILTER) {
      printf("On, ");
    } else {
      printf("Off, ");
    }
    printf("Auto ACK is ");
    if(value & RADIO_RX_MODE_AUTOACK) {
      printf("On, ");
    } else {
      printf("Off, ");
    }

    printf("(value=%d)\n", value);
  }

  printf("TX Mode: ");
  if(get_param(RADIO_PARAM_TX_MODE, &value) == RADIO_RESULT_OK) {
    printf("%d\n", value);
  }

  printf("TX Power: ");
  if(get_param(RADIO_PARAM_TXPOWER, &value) == RADIO_RESULT_OK) {
    printf("%d dBm [0x%04x]\n", value, (uint16_t)value);
  }

  printf("CCA Threshold: ");
  if(get_param(RADIO_PARAM_CCA_THRESHOLD, &value) == RADIO_RESULT_OK) {
    printf("%d dBm [0x%04x]\n", value, (uint16_t)value);
  }

  printf("RSSI: ");
  if(get_param(RADIO_PARAM_RSSI, &value) == RADIO_RESULT_OK) {
    printf("%d dBm [0x%04x]\n", value, (uint16_t)value);
  }
}
/*---------------------------------------------------------------------------*/

static int bytes2int(const unsigned char *bytes, unsigned char size) {
	int ret = 0; unsigned char i;
	for(i=0; i<size; i++) { ret <<= 8; ret |= bytes[i]; }
	return ret;
}

static void int2bytes(int num, unsigned char *bytes, unsigned char size) {
	while(size) { bytes[--size] = num & 0xFF; num >>= 8; }
}

static void uart_write_buffer(int begin, int size) {
  int i, end = begin + size;
  for (i = begin; i < end; i++) cc26xx_uart_write_byte(uart_buffer[i]);
}

static int sgetc(unsigned char data) {
  static int uart_size = 3, uart_count = 0;
  uart_buffer[uart_count++] = data;
  if (uart_count == 3 && uart_buffer[0] == TI_SEND) 
    uart_size = 3 + (bytes2int(uart_buffer+1, 2) << 6);
  if (uart_count == uart_size) {
    uart_write_buffer(0, uart_size);
    scmd_state = READY;
    uart_size = 3;
    uart_count = 0;
  } return 0;
}

static void receive() {
  if (!radio->read((void*)packet, PKTSIZE) || packet[9] != MAGIC) return;
  if (batch_id != bytes2int(packet+10, 2)) {
    batch_id = bytes2int(packet+10, 2);
    packet_count = 0;
  } packet_count ++;
  packet_id = bytes2int(packet+12, 2);
  memcpy(uart_buffer+3+(packet_id<<6), packet+14, 64);
  //uart_write_buffer(uart_size);
}

static void wz_zig() {
  static clock_time_t t; static int i, num;
  if (scmd_state != READY) return;
  scmd_state = WAIT;
  num = bytes2int(uart_buffer+1, 2);
  if (uart_buffer[0] == TI_TXPWR) {
    if(!radio_on) { TURN_RADIO(ON); radio_on = 1; }
    SET_TXPWR(num - 21);
  } else if (uart_buffer[0] == TI_RECV) {
    keep_radio_on = radio_on = 1;
    TURN_RADIO(ON);
    tasks[1] = &receive;
  } else if (uart_buffer[0] == TI_SEND) {
    if(!radio_on && num>0) TURN_RADIO(ON);
    packet[8] = PKTSIZE; packet[9] = MAGIC;
    int2bytes(++batch_id, packet+10, 2);
    t = clock_time();
    for (i=0; i < num; i++) {
      int2bytes(i, packet+12, 2);  // i is packet id
      memcpy(packet+14, uart_buffer+3+(i<<6), 64);
      while (radio->send(packet, PKTSIZE) != RADIO_TX_OK);
      if (clock_time() - t >= TX_TIMEOUT) break;
    } thruput = i * CLOCK_SECOND / (clock_time() - t);
    if(!keep_radio_on && radio_on) { TURN_RADIO(OFF); radio_on = 0; }
  } else if (uart_buffer[0] == TI_THPT) {
    int2bytes(thruput, uart_buffer+1, 2);
    uart_write_buffer(0,3);
  } else if (uart_buffer[0] == TI_PCNT) {
    int2bytes(packet_count, uart_buffer+1, 2);
    packet_count = 0;
    uart_write_buffer(0,3);
  } else if (uart_buffer[0] == TI_RXDAT) {
    uart_write_buffer(3, num << 6);
  }
}

static void loop_tasks(void (*tasks[]) (), clock_time_t delay) {
  int i;
  while(1) {
    if (delay) clock_wait(delay);
    for (i=0; i<N_TASK; ++i) if(tasks[i]) tasks[i]();
  }
}

PROCESS_THREAD(extended_rf_api_process, ev, data)
{
  PROCESS_BEGIN();

  get_rf_consts();
  print_rf_values();

  test_off_on();
  test_channels();
  test_rx_modes();
  test_tx_powers();
  test_cca_thresholds();

  printf("- - - start of experiment - - -\n");
  TURN_RADIO(OFF);
  cc26xx_uart_init();
  cc26xx_uart_set_input(sgetc);
  tasks[0] = &wz_zig;
  loop_tasks(tasks, 0);
  PROCESS_END();
}
/*---------------------------------------------------------------------------*/
