/* host-mock hardware/uart.h — uart_write_blocking captures bytes into the harness buffer. */
#pragma once
#include <stdint.h>
#include <stddef.h>

typedef struct mock_uart_inst uart_inst_t;
extern struct mock_uart_inst mock_uart1;
#define uart1 (&mock_uart1)
#define UART_PARITY_NONE 0

void uart_init(uart_inst_t *u, uint32_t baud);
void uart_set_format(uart_inst_t *u, int data, int stop, int parity);
void uart_write_blocking(uart_inst_t *u, const uint8_t *src, size_t len);
