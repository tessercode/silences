/*
 * enc_addons.c — OUTPUT-ONLY observers of the completion-spike emission (see enc_addons.h).
 *
 *   Feature 1 (ENC_AUDIO)     : Morse sidetone -> PCM5102 I2S DAC, on pio0 + one DMA channel, driven
 *                               entirely from CORE1. core0's emit_gpio() busy_wait spike loop is never
 *                               touched, so the GP6/GP7 edges E.2 measured are unchanged by construction.
 *   Feature 2 (ENC_CHAR_UART) : one ASCII byte per source character, 9600 8N1, on a monitor UART, also
 *                               emitted from CORE1 (never from inside the spike loop).
 *
 * THE SEAM (Step 0): the observers read the two things the emit loop already has —
 *   - g_tick[i] (absolute completion time, us) and g_ch[i] (0=dot line, 1=dash line), READ-ONLY, and
 *   - the cleaned `text`, walked through the PUBLIC, RNG-FREE Morse table (enc_morse_lookup) so each
 *     element's SOURCE CHARACTER and the character boundaries are recovered without touching the render.
 * No RNG is drawn (validation #3 holds), no render internal is called, no validated file is modified.
 *
 * AUDIO ON-DURATION FIDELITY (documented tradeoff): the completion-spike stream keeps only completion
 * times, so the pure on/off split is not on the wire. Each tone therefore ENDS at the real jittered
 * completion g_tick[i] (so tone SPACING carries the fist's omega-drift and gap jitter exactly) and has
 * the NOMINAL on-length (dot = 1*T_u, dash = r*T_u from the active wpm/r). The sub-element sigma_elem
 * on-time jitter (a few % of T_u, inaudible) is not separately voiced; recovering it would require
 * re-running the RNG, which is explicitly out of bounds.
 */
#include <stdint.h>
#include <string.h>
#include "pico/stdlib.h"
#include "encoder_render.h"        /* enc_morse_lookup() — public, RNG-free (reuse, don't fork) */

#if defined(ENC_AUDIO) || defined(ENC_CHAR_UART)
#include "pico/multicore.h"
#include "hardware/sync.h"         /* __dmb */
#endif

#ifdef ENC_CHAR_UART
#include "hardware/uart.h"
#include "hardware/gpio.h"
#endif

#ifdef ENC_AUDIO
#include <math.h>                  /* sinf, lrintf */
#include "hardware/pio.h"
#include "hardware/dma.h"
#include "hardware/irq.h"          /* DMA IRQ -> core1 sleeps on __wfe() (no APB polling during emit) */
#include "hardware/clocks.h"
#include "enc_i2s.pio.h"           /* generated from enc_i2s.pio (enc_audio_i2s_program*) */
#endif

#if defined(ENC_AUDIO) || defined(ENC_CHAR_UART)

/* ---------------------------------------------------------------------------------------------------
 * Shared alignment + capacities
 * ------------------------------------------------------------------------------------------------- */

/* g_tick's time origin is the FIRST element region, which emit_gpio reaches only AFTER it drives the
 * 400 ms held CAL + the 20 ms post-CAL gap (encoder_gpio.c: CAL_TICKS + POST_CAL_GAP_US). The observers
 * add this lead so tones/bytes land near the real spikes and the CAL window is audibly SILENT (the CAL
 * is a 400 ms held mark, NOT a Morse element — voicing it would sound like a very long dash). If those
 * two emit constants ever change, update this one line. Alignment need not be sample-exact: it does not
 * affect the validated GP6/GP7 edges, only how closely the sidetone tracks the wire. */
#define ADDON_CAL_LEAD_US   420000u

#define ADDON_MAX_ELEMS     256     /* == MAX_SPIKES in encoder_gpio.c */
#define ADDON_MAX_CHARS     512     /* >= any cleaned-text length (ENC_MAX_TEXT_LEN) */

/* The observer schedule: built once per transmission by core0 (in enc_addons_start), then consumed by
 * core1. core0 is blocked inside emit_gpio for the whole transmission, so it never rewrites this while
 * core1 reads it — the fifo push/pop around `seq` is the ordering barrier. */
typedef struct {
    uint32_t on_us[ADDON_MAX_ELEMS];    /* tone onset  (g_tick[i] - nominal on-length) */
    uint32_t off_us[ADDON_MAX_ELEMS];   /* tone offset (= g_tick[i], the real completion) */
    int      n_el;
    uint32_t char_us[ADDON_MAX_CHARS];  /* per-character: onset of the char's FIRST element */
    char     char_ascii[ADDON_MAX_CHARS];
    int      n_char;
    volatile uint32_t seq;              /* bumped each start */
} addon_sched_t;

static addon_sched_t g_sched;

static uint32_t nominal_on_us(char elem, double wpm, double r) {
    const double t_u = 1.2 / wpm;                          /* PARIS T_u; matches encoder_render.h */
    const double dur = (elem == '.') ? (1.0 * t_u) : (r * t_u);
    return (uint32_t)(dur * 1e6 + 0.5);
}

/* ---------------------------------------------------------------------------------------------------
 * Feature 2 — the character monitor UART (9600 8N1)
 *
 * ⚠️ THIS IS A MONITOR LINE, NOT A CHAIN LINK. It carries plain ASCII characters, NOT the 0x06 frame
 *    protocol; nothing downstream parses it. 8N1 = 8 data bits, No parity, 1 stop bit — matching the
 *    chain's frame links' line discipline, but the PAYLOAD is human-readable text, not framed bytes.
 * ------------------------------------------------------------------------------------------------- */
#ifdef ENC_CHAR_UART
#define CHAR_UART      uart0        /* FREE on this node: stdio_uart is disabled, no frame UART here */
#define CHAR_UART_TX   0u           /* GP0 = uart0 TX; no collision with GP6/7, GP18-20, GP23, USB, SWD */

static void char_uart_init(void) {
    uart_init(CHAR_UART, 9600);
    gpio_set_function(CHAR_UART_TX, GPIO_FUNC_UART);
    uart_set_format(CHAR_UART, 8, 1, UART_PARITY_NONE);   /* 8N1 */
    uart_set_fifo_enabled(CHAR_UART, true);               /* 32-byte TX FIFO -> enqueue is non-blocking */
}

/* Non-blocking enqueue. Chars are >= ~50 ms apart while a 9600 byte drains in ~1 ms and the FIFO is
 * 32 deep, so it never fills in practice; the writable check guarantees we never stall core1 (and core1
 * is not core0, so even a stall could not move a spike edge). */
static inline void char_uart_send(char c) {
    if (uart_is_writable(CHAR_UART)) uart_putc_raw(CHAR_UART, c);
}
#endif /* ENC_CHAR_UART */

/* ---------------------------------------------------------------------------------------------------
 * Feature 1 — the I2S sidetone (PCM5102)
 * ------------------------------------------------------------------------------------------------- */
#ifdef ENC_AUDIO
#define I2S_PIO          pio0       /* FREE: the identity LED is on pio1 (status_led.c) */
#define I2S_DATA_PIN     18u        /* DIN */
#define I2S_BCK_PIN      19u        /* BCK = clock_pin_base; LRCK/WS = base+1 = GP20 (consecutive) */
#define SAMPLE_RATE      44100u     /* 44.1 kHz, 16-bit stereo (BCK = 64*fs) */
#define SIDETONE_HZ      600u       /* CW sidetone convention (~600 Hz) */
#define AUDIO_AMPL       9830       /* ~0.30 of int16 full-scale; PCM5102 is LINE-LEVEL -> feed an amp /
                                     * powered speaker, NOT headphones directly */
#define RAMP_US          5000u      /* 5 ms linear key envelope (anti-click; a hard edge = key-clicks) */
#define AUDIO_BUF_FRAMES 256u       /* per ping-pong buffer (~5.8 ms @ 44.1 kHz) */
#define SINE_LUT_BITS    8
#define SINE_LUT_SIZE    (1u << SINE_LUT_BITS)

static PIO       s_pio = I2S_PIO;
static uint      s_sm;
static int       s_dma;
static uint32_t  audio_buf[2][AUDIO_BUF_FRAMES];   /* stereo frames: (L<<16)|R, L==R */
static int16_t   sine_lut[SINE_LUT_SIZE];
static const uint32_t phase_inc =
    (uint32_t)(((uint64_t)SIDETONE_HZ * SINE_LUT_SIZE << 16) / SAMPLE_RATE);   /* Q16 index step */

/* core1-owned playback state */
static volatile int s_active   = 0;
static uint64_t     s_frames   = 0;    /* total frames generated (free-running sample clock) */
static uint64_t     s_start    = 0;    /* s_frames at the current transmission's start */
static int          s_ai       = 0;    /* current audio element index */
static int          s_cur_el   = -1;   /* element whose sine phase is running (-1 = none) */
static uint32_t     s_phase    = 0;    /* Q16 sine phase */
#ifdef ENC_CHAR_UART
static int          s_ci       = 0;    /* current char index (UART, driven off the audio clock) */
#endif

static void audio_hw_init(void) {
    for (uint32_t i = 0; i < SINE_LUT_SIZE; i++)
        sine_lut[i] = (int16_t)lrintf(sinf(2.0f * 3.14159265358979f * (float)i / (float)SINE_LUT_SIZE)
                                      * (float)AUDIO_AMPL);
    uint off = pio_add_program(s_pio, &enc_audio_i2s_program);
    s_sm = (uint)pio_claim_unused_sm(s_pio, true);
    /* Connect the pins to PIO0 (the vendored program_init only sets pindirs; the GPIO function mux
     * lives in pico-extras' audio_i2s_setup, which we replicate here). Without this the SM clocks
     * internally but nothing reaches GP18/19/20. */
    pio_gpio_init(s_pio, I2S_DATA_PIN);        /* DIN  = GP18 */
    pio_gpio_init(s_pio, I2S_BCK_PIN);         /* BCK  = GP19 */
    pio_gpio_init(s_pio, I2S_BCK_PIN + 1u);    /* LRCK = GP20 (clock_pin_base + 1) */
    enc_audio_i2s_program_init(s_pio, s_sm, off, I2S_DATA_PIN, I2S_BCK_PIN);
    float div = (float)clock_get_hz(clk_sys) / (float)(SAMPLE_RATE * 64u);
    pio_sm_set_clkdiv(s_pio, s_sm, div);

    memset(audio_buf, 0, sizeof audio_buf);              /* start silent (zero samples, clock still runs) */
    s_dma = dma_claim_unused_channel(true);
    dma_channel_config c = dma_channel_get_default_config(s_dma);
    channel_config_set_transfer_data_size(&c, DMA_SIZE_32);
    channel_config_set_read_increment(&c, true);
    channel_config_set_write_increment(&c, false);
    channel_config_set_dreq(&c, pio_get_dreq(s_pio, s_sm, true));   /* TX DREQ, paces to fs */
    dma_channel_configure(s_dma, &c, &s_pio->txf[s_sm], audio_buf[0], AUDIO_BUF_FRAMES, false /*no trig*/);
    pio_sm_set_enabled(s_pio, s_sm, true);               /* BCK/LRCK begin; core1 keeps the FIFO fed */
}

/* Fill one buffer with the next AUDIO_BUF_FRAMES frames. Zero samples when idle -> the I2S clock keeps
 * running (PCM5102 needs a continuous clock; stopping it mutes/clicks). */
static void __not_in_flash_func(audio_fill)(uint32_t *buf) {   /* run from RAM: no XIP contention with core0 */
    for (uint32_t k = 0; k < AUDIO_BUF_FRAMES; k++) {
        int16_t s = 0;
        if (s_active) {
            uint32_t t_us = (uint32_t)(((s_frames - s_start) * 1000000ULL) / SAMPLE_RATE);
            while (s_ai < g_sched.n_el && t_us >= g_sched.off_us[s_ai] + ADDON_CAL_LEAD_US) s_ai++;
#ifdef ENC_CHAR_UART
            while (s_ci < g_sched.n_char && t_us >= g_sched.char_us[s_ci] + ADDON_CAL_LEAD_US)
                char_uart_send(g_sched.char_ascii[s_ci++]);
#endif
            if (s_ai < g_sched.n_el) {
                uint32_t on  = g_sched.on_us[s_ai]  + ADDON_CAL_LEAD_US;
                uint32_t off = g_sched.off_us[s_ai] + ADDON_CAL_LEAD_US;
                if (t_us >= on && t_us < off) {
                    if (s_ai != s_cur_el) { s_phase = 0; s_cur_el = s_ai; }   /* new element: phase reset */
                    uint32_t into = t_us - on, rem = off - t_us;
                    uint32_t e1 = into >= RAMP_US ? 32768u : (into * 32768u) / RAMP_US;
                    uint32_t e2 = rem  >= RAMP_US ? 32768u : (rem  * 32768u) / RAMP_US;
                    uint32_t env = e1 < e2 ? e1 : e2;                         /* Q15 raised edges */
                    int16_t base = sine_lut[(s_phase >> 16) & (SINE_LUT_SIZE - 1)];
                    s = (int16_t)(((int32_t)base * (int32_t)env) >> 15);
                    s_phase += phase_inc;
                }
            }
            if (s_ai >= g_sched.n_el
#ifdef ENC_CHAR_UART
                && s_ci >= g_sched.n_char
#endif
               ) s_active = 0;                                               /* transmission voiced */
        }
        uint16_t u = (uint16_t)s;
        buf[k] = ((uint32_t)u << 16) | u;                                    /* L == R */
        s_frames++;
    }
}

/* DMA-completion ISR, handled on CORE1 (see core1_main). Lets core1 SLEEP on __wfe() between buffers
 * instead of tight-polling the DMA status register over APB. THE INVARIANCE REWORK: that continuous APB
 * polling (dma_channel_wait_for_finish_blocking) contended with core0's busy_wait_us timer reads on the
 * shared APB bridge and jittered the GP6/GP7 spike edges (~2.5 us). With core1 asleep, the APB is core0's
 * alone during emit. emit_gpio is untouched. */
static volatile bool s_dma_done = false;
static void __not_in_flash_func(audio_dma_isr)(void) {   /* ISR in RAM: no XIP fetch on wake */
    if (dma_channel_get_irq0_status(s_dma)) {
        dma_channel_acknowledge_irq0(s_dma);
        s_dma_done = true;                    /* the IRQ itself also wakes core1's __wfe() */
    }
}
#endif /* ENC_AUDIO */

/* ---------------------------------------------------------------------------------------------------
 * core1 worker
 * ------------------------------------------------------------------------------------------------- */
static void __not_in_flash_func(core1_main)(void) {   /* core1's whole loop in RAM (XIP left to core0) */
#ifdef ENC_AUDIO
    /* Route DMA completion to THIS core's NVIC (core1) so the IRQ fires here, never on core0. */
    dma_channel_set_irq0_enabled(s_dma, true);
    irq_set_exclusive_handler(DMA_IRQ_0, audio_dma_isr);
    irq_set_enabled(DMA_IRQ_0, true);

    int play = 0;
    s_dma_done = false;
    dma_channel_transfer_from_buffer_now(s_dma, audio_buf[play], AUDIO_BUF_FRAMES);
    for (;;) {
        if (multicore_fifo_rvalid()) {           /* a new transmission was handed over */
            (void)multicore_fifo_pop_blocking();
            s_start = s_frames; s_ai = 0; s_cur_el = -1; s_phase = 0;
#ifdef ENC_CHAR_UART
            s_ci = 0;
#endif
            s_active = 1;
        }
        audio_fill(audio_buf[play ^ 1]);         /* fill the buffer that plays NEXT (short RAM burst) */
        while (!s_dma_done) __wfe();              /* SLEEP until the buffer drains — no APB polling, so
                                                    core0's busy_wait_us timer reads are uncontended */
        s_dma_done = false;
        play ^= 1;
        dma_channel_transfer_from_buffer_now(s_dma, audio_buf[play], AUDIO_BUF_FRAMES);
    }
#elif defined(ENC_CHAR_UART)
    /* UART-ONLY build (audio OFF): core1 waits for each char's time with a light timer poll.
     * ⚠️ KNOWN LIMITATION — spike-edge invariance: this tight-ish time_us_64() poll hammers the APB
     * bridge and WILL jitter core0's spikes the same way the audio DMA-status poll did. This standalone
     * build is NOT the validated config (the deployed image runs audio ON, which uses the DMA-IRQ/__wfe
     * path above and passes the gate). If char-UART-only is ever deployed with strict spike timing, give
     * it the same treatment: wake core1 from a hardware-alarm IRQ on THIS core + __wfe(), not a poll. */
    for (;;) {
        (void)multicore_fifo_pop_blocking();      /* wait for a transmission */
        uint64_t t0 = time_us_64();
        int ci = 0;
        while (ci < g_sched.n_char) {
            if (time_us_64() - t0 >= (uint64_t)g_sched.char_us[ci] + ADDON_CAL_LEAD_US)
                char_uart_send(g_sched.char_ascii[ci++]);
            else
                tight_loop_contents();
        }
    }
#endif
}

/* ---------------------------------------------------------------------------------------------------
 * Public API
 * ------------------------------------------------------------------------------------------------- */
void enc_addons_init(void) {
    g_sched.seq = 0; g_sched.n_el = 0; g_sched.n_char = 0;
#ifdef ENC_CHAR_UART
    char_uart_init();
#endif
#ifdef ENC_AUDIO
    audio_hw_init();
#endif
    multicore_launch_core1(core1_main);
}

void enc_addons_start(const char *text, const uint32_t *tick, const uint8_t *ch,
                      int n, double wpm, double r) {
    (void)ch;                                    /* on-length comes from the Morse element, not the line */
    int ei = 0, ne = 0, nc = 0;
    uint32_t prev_off = 0;
    for (const char *p = text; *p; ++p) {
        char c = *p;
        if (c == ' ') {                          /* inter-word space -> monitor a space (decision #18) */
            if (nc < ADDON_MAX_CHARS) { g_sched.char_us[nc] = prev_off; g_sched.char_ascii[nc] = ' '; nc++; }
            continue;
        }
        const char *els = enc_morse_lookup(c);
        if (!els) continue;                      /* text is already cleaned; defensive */
        int char_started = 0;
        for (int e = 0; els[e]; ++e) {
            if (ei >= n) break;                  /* respect the emitted count (cap guard) */
            uint32_t off   = tick[ei];
            uint32_t onlen = nominal_on_us(els[e], wpm, r);
            uint32_t on    = (off > onlen) ? (off - onlen) : 0u;
            if (on < prev_off) on = prev_off;    /* never overlap the previous tone */
            if (ne < ADDON_MAX_ELEMS) { g_sched.on_us[ne] = on; g_sched.off_us[ne] = off; ne++; }
            if (!char_started) {                 /* emit the char at its FIRST element's onset */
                if (nc < ADDON_MAX_CHARS) { g_sched.char_us[nc] = on; g_sched.char_ascii[nc] = c; nc++; }
                char_started = 1;
            }
            prev_off = off;
            ei++;
        }
    }
    g_sched.n_el = ne;
    g_sched.n_char = nc;
    __dmb();                                     /* publish schedule before the wake */
    g_sched.seq++;
    multicore_fifo_push_blocking(g_sched.seq);   /* barrier + wake core1 */
}

#else  /* neither feature: keep the TU legal but empty (it is not compiled into the shipping target) */
void enc_addons_init(void) {}
void enc_addons_start(const char *text, const uint32_t *tick, const uint8_t *ch,
                      int n, double wpm, double r) {
    (void)text; (void)tick; (void)ch; (void)n; (void)wpm; (void)r;
}
#endif
