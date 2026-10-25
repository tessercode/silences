/* B2b.1 host harness — MOCK hardware/pio.h. The PIO receiver is a transport the host does not
 * exercise: the harness feeds bytes straight into the SHIPPING rx_feed_byte(), so the assembler
 * (fa_feed) is under test while the FIFO is not. */
#ifndef MOCK_HARDWARE_PIO_H
#define MOCK_HARDWARE_PIO_H
#include <stdint.h>
#include <stdbool.h>
typedef struct { int id; } *PIO;
typedef unsigned uint;
#define pio0 ((PIO)0)
#define pio1 ((PIO)1)
enum pio_interrupt_source { pis_sm0_rx_fifo_not_empty = 0 };
static inline void pio_set_irq0_source_enabled(PIO p, enum pio_interrupt_source s, bool e) {
    (void)p; (void)s; (void)e;
}
static inline int  pio_claim_unused_sm(PIO p, bool r) { (void)p; (void)r; return 0; }
static inline uint pio_add_program(PIO p, const void *prog) { (void)p; (void)prog; return 0; }
#endif
