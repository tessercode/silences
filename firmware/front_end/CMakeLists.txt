# The front-end board: captures the encoder's two channel lines and sends one 0x06 frame per transmission.
cmake_minimum_required(VERSION 3.13)
include(pico_sdk_import.cmake)
project(fe_sentence_b1 C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(PICO_BOARD pico)
set(PICO_PLATFORM rp2040)
pico_sdk_init()

add_executable(fe_sentence_b1 frontend_stage3.c frontend.c frontend_led.c status_led.c)
target_include_directories(fe_sentence_b1 PRIVATE ${CMAKE_CURRENT_LIST_DIR})
pico_generate_pio_header(fe_sentence_b1 ${CMAKE_CURRENT_LIST_DIR}/capture.pio)
pico_generate_pio_header(fe_sentence_b1 ${CMAKE_CURRENT_LIST_DIR}/ws2812.pio)
target_link_libraries(fe_sentence_b1 pico_stdlib hardware_pio hardware_uart)

# Stack: est_shortest_isi keeps a stack-local int64 isis[FE_MAXN], 2 KB at FE_MAXN = 256, which alone
# fills the SDK's default 0x800. 0x1000 gives the estimator and its call chain about 2x headroom; the
# large frame buffers are static.
target_compile_definitions(fe_sentence_b1 PRIVATE PICO_STACK_SIZE=0x1000)

pico_enable_stdio_uart(fe_sentence_b1 1)
pico_enable_stdio_usb(fe_sentence_b1 0)
pico_add_extra_outputs(fe_sentence_b1)
