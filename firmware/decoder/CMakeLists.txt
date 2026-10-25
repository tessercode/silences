# The decoder board: 0x05 frames in from the router, the integer beam search with the Q16.16 character
# language model, decoded text out over USB.
#
# Targets:
#   decode_board   CHARLM_EMBED=ON (default): the image of record. OFF builds the USB bridge alone.
#   b2b3_test      the emulator test: baked 0x05 frames in, decoded result lines out on uart0.
cmake_minimum_required(VERSION 3.13)
include(pico_sdk_import.cmake)
project(decode_board C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(PICO_BOARD pico)
set(PICO_PLATFORM rp2040)
pico_sdk_init()

option(CHARLM_EMBED "embed the char-LM table + lookup + beam" ON)

set(SRCS output_board.c status_led.c)
if(CHARLM_EMBED)
    list(APPEND SRCS charlm_lookup.c beam.c)
endif()

add_executable(decode_board ${SRCS})
pico_generate_pio_header(decode_board ${CMAKE_CURRENT_LIST_DIR}/ws2812.pio)
target_include_directories(decode_board PRIVATE ${CMAKE_CURRENT_LIST_DIR}
                           ${CMAKE_CURRENT_LIST_DIR}/../common)   # shared router_core.h
target_compile_options(decode_board PRIVATE -fstack-usage)
target_link_libraries(decode_board pico_stdlib hardware_uart hardware_pio)
if(CHARLM_EMBED)
    target_compile_definitions(decode_board PRIVATE CHARLM_EMBED=1)
endif()
# bench option: record the receive ring's peak occupancy, readable over SWD (frame path unchanged)
if(B2_SILICON_HWM)
    target_compile_definitions(decode_board PRIVATE RX_RING_TRACK_HWM=1)
endif()

pico_enable_stdio_uart(decode_board 0)     # UART pins are the router link (as the original)
pico_enable_stdio_usb(decode_board 1)      # USB-CDC relay (as the original)
pico_add_extra_outputs(decode_board)

# ---- emulator test: baked 0x05 frames in -> decoded result lines out on uart0 ----
# Renode models only uart0 and cannot drive a UART RX line, so the runner injects at rx_feed_byte
# and pins stdio to uart0. Transport only; the decoded CONTENT is what the comparator gates.
# n1/n5 differ only in the baked posteriors (rebake with gen_b2b3_vectors.py + rebuild).
add_executable(b2b3_test main_b2b3_test.c beam.c charlm_lookup.c status_led.c)
pico_generate_pio_header(b2b3_test ${CMAKE_CURRENT_LIST_DIR}/ws2812.pio)
target_include_directories(b2b3_test PRIVATE ${CMAKE_CURRENT_LIST_DIR}
                           ${CMAKE_CURRENT_LIST_DIR}/../common)
target_compile_definitions(b2b3_test PRIVATE CHARLM_EMBED=1)
target_compile_options(b2b3_test PRIVATE -fstack-usage)
target_link_libraries(b2b3_test pico_stdlib hardware_uart hardware_pio)
pico_enable_stdio_uart(b2b3_test 1)      # uart0: the result lines (Renode captures this)
pico_enable_stdio_usb(b2b3_test 0)
pico_add_extra_outputs(b2b3_test)
