cmake_minimum_required(VERSION 3.13)

include(pico_sdk_import.cmake)

project(encoder C CXX ASM)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

set(PICO_BOARD pico)
set(PICO_PLATFORM rp2040)

pico_sdk_init()

set(COMMON_DIR ${CMAKE_CURRENT_LIST_DIR}/../common)

# ---- the encoder board: keys a sentence from the console and emits completion spikes on GPIO ----
add_executable(encoder_gpio encoder_gpio.c encoder_led.c ${COMMON_DIR}/status_led.c)
pico_generate_pio_header(encoder_gpio ${COMMON_DIR}/ws2812.pio)
target_include_directories(encoder_gpio PRIVATE ${CMAKE_CURRENT_LIST_DIR} ${COMMON_DIR})
target_compile_options(encoder_gpio PRIVATE -fstack-usage)
target_link_libraries(encoder_gpio pico_stdlib hardware_pio)
# Stack: the deepest chain (measured with -fstack-usage) is main -> process_line -> enc_render_text ->
# encoder_render_sentence_jit -> enc_render_word_core_jit -> enc_rng_normal. The large buffers are
# static, not stack frames.
target_compile_definitions(encoder_gpio PRIVATE PICO_STACK_SIZE=0x1000)
pico_enable_stdio_uart(encoder_gpio 0)
pico_enable_stdio_usb(encoder_gpio 1)           # typed-sentence console
pico_add_extra_outputs(encoder_gpio)

# ---- optional outputs (enc_addons): a Morse sidetone and a character-monitor UART ----
# Both off by default, and then nothing below runs and encoder_gpio is the image of record. Enable with:
#     cmake -DENC_AUDIO=ON            (I2S sidetone only)
#     cmake -DENC_CHAR_UART=ON        (char monitor UART only)
#     cmake -DENC_AUDIO=ON -DENC_CHAR_UART=ON   (both)
option(ENC_AUDIO     "Feature 1: Morse sidetone on a PCM5102 I2S DAC (pio0 + DMA + core1)" OFF)
option(ENC_CHAR_UART "Feature 2: one ASCII byte per character on a monitor UART (9600 8N1)" OFF)
if(ENC_AUDIO OR ENC_CHAR_UART)
    target_sources(encoder_gpio PRIVATE enc_addons.c)
    target_link_libraries(encoder_gpio pico_multicore)   # observers run on core1
    if(ENC_CHAR_UART)
        target_compile_definitions(encoder_gpio PRIVATE ENC_CHAR_UART)
        target_link_libraries(encoder_gpio hardware_uart)
    endif()
    if(ENC_AUDIO)
        target_compile_definitions(encoder_gpio PRIVATE ENC_AUDIO)
        target_link_libraries(encoder_gpio hardware_pio hardware_dma hardware_clocks)
        pico_generate_pio_header(encoder_gpio ${CMAKE_CURRENT_LIST_DIR}/enc_i2s.pio)
    endif()
    message(STATUS "enc_addons: ENC_AUDIO=${ENC_AUDIO} ENC_CHAR_UART=${ENC_CHAR_UART}")
endif()

# ---- emulator test: baked inputs in -> the computed tick/channel stream out on uart0 ----
# Renode's RP2040 models only uart0 and cannot observe GPIO, so the test reports what would be emitted
# over uart0 and the comparator checks it. The GPIO emission itself is checked only on the bench.
add_executable(e1_test main_e1_test.c encoder_led.c ${COMMON_DIR}/status_led.c)
pico_generate_pio_header(e1_test ${COMMON_DIR}/ws2812.pio)
target_include_directories(e1_test PRIVATE ${CMAKE_CURRENT_LIST_DIR} ${COMMON_DIR})
target_compile_options(e1_test PRIVATE -fstack-usage)
target_link_libraries(e1_test pico_stdlib hardware_pio)
target_compile_definitions(e1_test PRIVATE PICO_STACK_SIZE=0x1000)
pico_enable_stdio_uart(e1_test 1)               # uart0 carries the tick stream ALONE
pico_enable_stdio_usb(e1_test 0)
pico_add_extra_outputs(e1_test)
