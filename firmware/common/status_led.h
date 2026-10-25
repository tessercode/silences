/*
 * status_led.h — SHARED status-LED module for the TEMPO RP2040 pipeline boards.
 *
 * Layer 1 (this file): a STATIC IDENTITY COLOR set once at boot, so each board announces its firmware
 * type by colour. Callers deal only in plain module identities — the WS2812 GRB byte order and the
 * global brightness scale are encapsulated in status_led.c; callers never touch either.
 *
 * The API is shaped so Layers 2/3 (status states, non-blocking pulse/animation) drop in later WITHOUT
 * changing existing call sites — see the extension-points block at the bottom.
 */
#ifndef STATUS_LED_H
#define STATUS_LED_H

/* Board firmware identities. Keep MODULE__COUNT last (array sizing). DECODE is a future board. */
typedef enum {
    MODULE_ROUTER = 0,
    MODULE_ENCODER,
    MODULE_FRONTEND,
    MODULE_GAP9,
    MODULE_DECODE,        /* future */
    MODULE_GAP9_LIF,      /* spiking_gap9_lif drop-in (LIF segmenter) — distinct from MODULE_GAP9 */
    MODULE_GAP9_MLP,      /* spiking_gap9_mlp drop-in (tanh MLP segmenter) */
    MODULE__COUNT
} module_id_t;

/* Layer 1 API — this is all a board needs at boot. */
void status_led_init(void);                         /* PIO/pin setup on the onboard WS2812; LED off */
void status_led_set_identity(module_id_t module);   /* solid identity colour (see table in .c) */

/* ---------------------------------------------------------------------------------------------------
 * Layer 2/3 EXTENSION POINTS — shapes only; NOT implemented in Layer 1. Documented here so a later
 * animation/status layer is added WITHOUT touching identity or any call site:
 *
 *   typedef enum { STATUS_OK, STATUS_BUSY, STATUS_FAULT } status_state_t;   // Layer 2
 *   void status_led_set_state(status_state_t st);   // overlay a state on the identity (FAULT -> RED,
 *                                                   //   the colour RESERVED in .c and never an identity)
 *   void status_led_tick(void);                     // Layer 3: non-blocking animation step, called from
 *                                                   //   the board's main loop (pulse/blink). No delays.
 *
 * The .c already keeps the current pixel colour in a static so an animation layer can read/restore it;
 * identity remains the base layer these compose over.
 * ------------------------------------------------------------------------------------------------- */

#endif /* STATUS_LED_H */
