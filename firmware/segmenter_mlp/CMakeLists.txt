cmake_minimum_required(VERSION 3.13)
include(pico_sdk_import.cmake)
project(spiking_gap9_mlp C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(PICO_BOARD pico)
set(PICO_PLATFORM rp2040)
pico_sdk_init()

# The tanh MLP gap segmenter. Only clf_forward.c + clf_weights.h are this module's; the compute
# core, transport, bench, LUTs and vectors are SHARED in ../common/ (see common/clf_core.h).
set(COMMON ${CMAKE_CURRENT_LIST_DIR}/../common)

# Renode/host bench: baked vectors -> clf_run -> "vid ng lp..." on uart0.
add_executable(clf_bench ${COMMON}/clf_bench_main.c clf_forward.c)
target_include_directories(clf_bench PRIVATE ${CMAKE_CURRENT_LIST_DIR} ${COMMON})
target_compile_options(clf_bench PRIVATE -fstack-usage)
target_link_libraries(clf_bench pico_stdlib)
pico_enable_stdio_uart(clf_bench 1)
pico_enable_stdio_usb(clf_bench 0)
pico_add_extra_outputs(clf_bench)

# Bench latency build: identical to clf_bench but stdio on uart1 (TX GP4, RX GP5, 115200), the pin the
# NI DAQ taps (bench/latency_test.py). Compute untouched.
add_executable(clf_bench_gp4 ${COMMON}/clf_bench_main.c clf_forward.c)
target_include_directories(clf_bench_gp4 PRIVATE ${CMAKE_CURRENT_LIST_DIR} ${COMMON})
target_compile_options(clf_bench_gp4 PRIVATE -fstack-usage)
target_link_libraries(clf_bench_gp4 pico_stdlib)
target_compile_definitions(clf_bench_gp4 PRIVATE
    PICO_DEFAULT_UART=1 PICO_DEFAULT_UART_TX_PIN=4 PICO_DEFAULT_UART_RX_PIN=5)
pico_enable_stdio_uart(clf_bench_gp4 1)
pico_enable_stdio_usb(clf_bench_gp4 0)
pico_add_extra_outputs(clf_bench_gp4)

# Deployment drop-in for the gap9 board: live 0x06 -> clf_run_marks -> 0x03 (shared transport).
add_executable(clf_hw ${COMMON}/clf_hw.c clf_forward.c mlp_led.c ${COMMON}/status_led.c)  # white identity
pico_generate_pio_header(clf_hw ${COMMON}/uart_rx.pio)
pico_generate_pio_header(clf_hw ${COMMON}/ws2812.pio)
target_include_directories(clf_hw PRIVATE ${CMAKE_CURRENT_LIST_DIR} ${COMMON})
target_compile_options(clf_hw PRIVATE -fstack-usage)
target_link_libraries(clf_hw pico_stdlib hardware_uart hardware_pio)
target_compile_definitions(clf_hw PRIVATE PICO_STACK_SIZE=0x1000 RX_RING_TRACK_HWM=1)  # receive-ring peak occupancy, readable over SWD
pico_enable_stdio_uart(clf_hw 1)
pico_enable_stdio_usb(clf_hw 0)
pico_add_extra_outputs(clf_hw)
