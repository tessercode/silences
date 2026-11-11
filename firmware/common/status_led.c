/*
 * status_led.c — SHARED status-LED module (Layer 1: static identity colour).
 *
 * Drives the YD-RP2040 onboard RGB — a single WS2812 on GP23 — via the standard Pico SDK ws2812 PIO
 * program (common/ws2812.pio). Encapsulates, so callers never see them:
 *   - GRB byte order (WS2812 wants green-red-blue; callers pass plain 0xRRGGBB),
 *   - a modest global brightness scale (~15% of full — identifiable, not blinding).
 *
 * PIO ALLOCATION (the one integration risk on a board that already uses PIO): this module uses a
 * DYNAMICALLY-CLAIMED unused SM on **pio1**. The router's UART receivers live on **pio0** (SM0/SM1),
 * so there is no state-machine or instruction-memory contention. Any board whose own PIO use is on
 * pio0 is likewise safe; a board already using pio1 still gets a free SM via pio_claim_unused_sm.
 */
#include "status_led.h"
#include "pico/stdlib.h"
#include "hardware/pio.h"
#include "ws2812.pio.h"

/* ---- hardware config ---- */
#define LED_PIN         23u        /* YD-RP2040 onboard WS2812 (confirmed) */
#define LED_PIO         pio1       /* isolated from the boards' UART-RX PIO (pio0) */
#define LED_BRIGHTNESS_PCT  6u     /* global scale: ~6% of full (WS2812 is bright; identity, not a lamp) */

/* ---- packed colour helpers ---- */
#define RGB(r, g, b)  (((uint32_t)(r) << 16) | ((uint32_t)(g) << 8) | (uint32_t)(b))

/* Named colours. RED is RESERVED for error/fault (Layer 2+) so "red = problem" is a rig-wide
 * convention — it is deliberately NOT assigned as any board's identity below. */
#define COL_CYAN     RGB(0x00, 0xFF, 0xFF)
#define COL_GREEN    RGB(0x00, 0xFF, 0x00)
#define COL_BLUE     RGB(0x00, 0x00, 0xFF)
#define COL_MAGENTA  RGB(0xFF, 0x00, 0xFF)
#define COL_YELLOW   RGB(0xFF, 0xFF, 0x00)
#define COL_RED      RGB(0xFF, 0x00, 0x00)   /* RESERVED: fault only; never an identity */
#define COL_ORANGE   RGB(0xFF, 0x45, 0x00)   /* spiking_gap9_lif identity (distinct from yellow/red) */
#define COL_WHITE    RGB(0xFF, 0xFF, 0xFF)   /* spiking_gap9_mlp identity (distinct, desaturated) */
#define COL_VIOLET   RGB(0x50, 0x00, 0xFF)   /* spiking_gap9_spk identity (between blue and magenta) */
#define COL_OFF      RGB(0x00, 0x00, 0x00)

/* IDENTITY COLOUR SCHEME — single source of truth; change colours here in one place. */
static const uint32_t IDENTITY_COLOR[MODULE__COUNT] = {
    [MODULE_ROUTER]   = COL_CYAN,
    [MODULE_ENCODER]  = COL_GREEN,
    [MODULE_FRONTEND] = COL_BLUE,
    [MODULE_GAP9]     = COL_MAGENTA,
    [MODULE_DECODE]   = COL_YELLOW,
    [MODULE_GAP9_LIF] = COL_ORANGE,
    [MODULE_GAP9_MLP] = COL_WHITE,
    [MODULE_GAP9_SPK] = COL_VIOLET,
};

/* ---- module state ---- */
static PIO      s_pio;
static uint     s_sm;
static bool     s_ready = false;
static uint32_t s_current_rgb = COL_OFF;   /* last shown colour — a Layer 2/3 animation layer can
                                              read/restore this; identity is the base it composes over */

static inline uint8_t scale8(uint8_t v) {  /* integer global-brightness scale */
    return (uint8_t)(((uint32_t)v * LED_BRIGHTNESS_PCT) / 100u);
}

/* Encapsulated push: apply brightness, convert RGB -> GRB, send 24 bits (left-justified for autopull). */
static void led_show(uint32_t rgb) {
    if (!s_ready) return;
    uint8_t r = scale8((uint8_t)(rgb >> 16));
    uint8_t g = scale8((uint8_t)(rgb >> 8));
    uint8_t b = scale8((uint8_t)(rgb));
    uint32_t grb = ((uint32_t)g << 16) | ((uint32_t)r << 8) | (uint32_t)b;   /* WS2812 GRB order */
    pio_sm_put_blocking(s_pio, s_sm, grb << 8u);                             /* 24 data bits in the MSBs */
    /* WS2812 LATCH: hold the line idle-low > reset time so THIS frame latches before any next one.
     * Without it, two back-to-back updates (init OFF, then identity) merge into a 2-pixel stream and a
     * single LED latches only the first (OFF) -> stays dark. ~300us covers the ~30us shift + >50us reset.
     * (A future non-blocking Layer 3 must instead space frames by time; see status_led.h.) */
    sleep_us(300);
    s_current_rgb = rgb;
}

void status_led_init(void) {
    s_pio = LED_PIO;
    int sm = pio_claim_unused_sm(s_pio, true);        /* claim a free SM on pio1 (panics if none) */
    s_sm  = (uint)sm;
    uint offset = pio_add_program(s_pio, &ws2812_program);
    ws2812_program_init(s_pio, s_sm, offset, LED_PIN);
    s_ready = true;
    led_show(COL_OFF);                                /* start dark until an identity is set */
}

void status_led_set_identity(module_id_t module) {
    if (module >= MODULE__COUNT) return;
    led_show(IDENTITY_COLOR[module]);
}
