/* B2a host harness — MOCK hardware/irq.h. The ISR is not exercised on host (that needs real
 * hardware); the harness drives the ring's PRODUCER directly, so arming is a no-op here.
 * ⚠️ This is exactly the coverage statement in b2a_port.md §5: the ISR FIRING is silicon's. */
#ifndef MOCK_HARDWARE_IRQ_H
#define MOCK_HARDWARE_IRQ_H
#include <stdint.h>
#define PIO0_IRQ_0 7
#define UART1_IRQ  21
typedef void (*irq_handler_t)(void);
static inline void irq_set_exclusive_handler(unsigned n, irq_handler_t h) { (void)n; (void)h; }
static inline void irq_set_enabled(unsigned n, int e) { (void)n; (void)e; }
#endif
