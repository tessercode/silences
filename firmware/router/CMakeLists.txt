cmake_minimum_required(VERSION 3.13)
include(pico_sdk_import.cmake)
project(router_hw C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(PICO_BOARD pico)          # YD-RP2040 board; GP23 RGB / GP24 button avoided (as the other modules)
set(PICO_PLATFORM rp2040)
pico_sdk_init()

# Interconnect-router firmware: two PIO UART receivers + the shared portable core (router_core.h),
# plus the SHARED status-LED identity module (common/status_led.c + common/ws2812.pio).
set(COMMON_DIR ${CMAKE_CURRENT_LIST_DIR}/../common)
add_executable(router_hw router_hw.c ${COMMON_DIR}/status_led.c)
pico_generate_pio_header(router_hw ${COMMON_DIR}/uart_rx.pio)   # shared PIO UART receiver
pico_generate_pio_header(router_hw ${COMMON_DIR}/ws2812.pio)                # -> ws2812.pio.h
target_include_directories(router_hw PRIVATE ${CMAKE_CURRENT_LIST_DIR} ${COMMON_DIR})
target_link_libraries(router_hw pico_stdlib hardware_pio hardware_uart)
# bench option: record the receive rings' peak occupancy, readable over SWD (frame path unchanged)
if(B2_SILICON_HWM)
    target_compile_definitions(router_hw PRIVATE RX_RING_TRACK_HWM=1)
endif()
pico_enable_stdio_uart(router_hw 1)     # uart0 GP0/GP1 debug sentinel; forward line is uart1 on GP4
pico_enable_stdio_usb(router_hw 0)
# Stack: the large staging buffers (9,984 B at the 256-mark cap) are static, not locals; 0x1000 keeps
# the same >= 25% headroom rule as the other boards.
target_compile_definitions(router_hw PRIVATE PICO_STACK_SIZE=0x1000)
target_compile_options(router_hw PRIVATE -fstack-usage)
pico_add_extra_outputs(router_hw)

# ---- emulator test: baked 0x06 + 0x03 frames in -> 0x05 frames out ----
# Renode models only uart0 and cannot drive the two PIO receive lines, so the test feeds the baked frames
# through the shipping core directly and sends 0x05 on uart0. The receive path itself is checked only on
# the bench.
add_executable(rtr_test main_rtr_test.c ${COMMON_DIR}/status_led.c)
pico_generate_pio_header(rtr_test ${COMMON_DIR}/uart_rx.pio)
pico_generate_pio_header(rtr_test ${COMMON_DIR}/ws2812.pio)
target_include_directories(rtr_test PRIVATE ${CMAKE_CURRENT_LIST_DIR} ${COMMON_DIR})
target_compile_definitions(rtr_test PRIVATE PICO_STACK_SIZE=0x1000
                           FWD_UART=uart0 PIN_FWD=0u)
target_compile_options(rtr_test PRIVATE -fstack-usage)
target_link_libraries(rtr_test pico_stdlib hardware_pio hardware_uart)
pico_enable_stdio_uart(rtr_test 0)      # uart0 carries the 0x05 stream ALONE
pico_enable_stdio_usb(rtr_test 0)
pico_add_extra_outputs(rtr_test)
