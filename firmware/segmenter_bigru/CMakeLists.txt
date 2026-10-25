cmake_minimum_required(VERSION 3.13)

include(pico_sdk_import.cmake)

project(gap9_3c C CXX ASM)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

set(PICO_BOARD pico)
set(PICO_PLATFORM rp2040)

pico_sdk_init()

# ---- the BiGRU segmenter board: 0x06 frames in on PIO UART, 0x03 frames out ----
# n1 and n5 are the same code built with -DGAP9_3C_N_SEEDS=1 or 5; only the seed loop differs.
foreach(ns 1 5)
    add_executable(gap9_hw_3c_n${ns} gap9_hw.c gap9_3c.c gap9_led.c status_led.c)
    pico_generate_pio_header(gap9_hw_3c_n${ns} ${CMAKE_CURRENT_LIST_DIR}/../common/uart_rx.pio)
    pico_generate_pio_header(gap9_hw_3c_n${ns} ${CMAKE_CURRENT_LIST_DIR}/ws2812.pio)
    target_include_directories(gap9_hw_3c_n${ns} PRIVATE ${CMAKE_CURRENT_LIST_DIR}
                               ${CMAKE_CURRENT_LIST_DIR}/../common)   # shared router_core.h
    target_compile_options(gap9_hw_3c_n${ns} PRIVATE -fstack-usage)
    target_compile_definitions(gap9_hw_3c_n${ns} PRIVATE GAP9_3C_N_SEEDS=${ns})
    # bench option: record the receive ring's peak occupancy, readable over SWD (frame path unchanged)
    if(B2_SILICON_HWM)
        target_compile_definitions(gap9_hw_3c_n${ns} PRIVATE RX_RING_TRACK_HWM=1)
    endif()
    target_link_libraries(gap9_hw_3c_n${ns} pico_stdlib hardware_uart hardware_pio)
    # Stack: the deepest chain (-fstack-usage) is main(112) -> gap9_run_marks(8) -> gap9_finish(168)
    # -> gru_dir(1184) = 1,472 B, plus 80 B for interrupts. 0x1000 leaves 62% headroom (the SDK
    # default 0x800 would leave 24%). The large buffers are static.
    target_compile_definitions(gap9_hw_3c_n${ns} PRIVATE PICO_STACK_SIZE=0x1000)
    # stdio on uart0/GP0 (Renode captures it); the 0x03 result line is uart1/GP4 as on the board.
    pico_enable_stdio_uart(gap9_hw_3c_n${ns} 1)
    pico_enable_stdio_usb(gap9_hw_3c_n${ns} 0)
    pico_add_extra_outputs(gap9_hw_3c_n${ns})
endforeach()

# ---- emulator test: baked 0x06 frames in -> 0x03 frames out (the comparator checks them) ----
foreach(ns 1 5)
    add_executable(b2b_test_n${ns} main_b2b_test.c gap9_3c.c gap9_led.c status_led.c)
    pico_generate_pio_header(b2b_test_n${ns} ${CMAKE_CURRENT_LIST_DIR}/../common/uart_rx.pio)
    pico_generate_pio_header(b2b_test_n${ns} ${CMAKE_CURRENT_LIST_DIR}/ws2812.pio)
    target_include_directories(b2b_test_n${ns} PRIVATE ${CMAKE_CURRENT_LIST_DIR}
                               ${CMAKE_CURRENT_LIST_DIR}/../common)   # shared router_core.h
    target_compile_options(b2b_test_n${ns} PRIVATE -fstack-usage)
    target_compile_definitions(b2b_test_n${ns} PRIVATE GAP9_3C_N_SEEDS=${ns})
    target_link_libraries(b2b_test_n${ns} pico_stdlib hardware_uart hardware_pio)
    # Re-point the 0x03 line to uart0/GP0: the Renode RP2040 platform models only uart0
    # (cores/rp2040.repl:47). Transport only — the frame bytes are identical.
    target_compile_definitions(b2b_test_n${ns} PRIVATE PICO_STACK_SIZE=0x1000
                               RESULT_UART=uart0 PIN_RESULT=0u)
    pico_enable_stdio_uart(b2b_test_n${ns} 0)   # uart0 carries the 0x03 stream ALONE
    pico_enable_stdio_usb(b2b_test_n${ns} 0)
    pico_add_extra_outputs(b2b_test_n${ns})
endforeach()
