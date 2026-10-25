# Third-party material

Everything here is MIT-licensed (see `LICENSE`) except the following, which keep their own terms.

| Item | Where | Licence | Notes |
|---|---|---|---|
| xoshiro128** 1.1 and SplitMix64 reference code (Blackman and Vigna) | `firmware/tests/kat/reference/` | Public domain (CC0) | Unmodified; sha256 in `firmware/tests/kat/provenance.json` |
| `pico_sdk_import.cmake` | each firmware module | BSD-3-Clause (Raspberry Pi Ltd) | From the Pico SDK; header retained |
| `uart_rx.pio`, `ws2812.pio` | `firmware/` | BSD-3-Clause (Raspberry Pi Ltd) | Adapted from the pico-examples programs of the same name |
| Raspberry Pi Pico SDK 2.2.0 | fetched by `firmware/third_party_fetch.sh` | BSD-3-Clause | Build dependency, not included |
| Renode 1.16.1, matgla/Renode_RP2040 | fetched by `firmware/third_party_fetch.sh` | MIT | Emulator tests only, not included |
| NLTK `words` and `brown` corpora | downloaded by `make setup` | see the NLTK data licences | English vocabulary and the language model's English text, not included |

## The real recordings

The five synthetic operators (`firmware/encoder/fists.h`) were fitted to 44 amateur-radio recordings,
used with the written permission of their archive owners. The recordings are not redistributed. Their
per-segment fits are in `data/operator_style_fits.csv`, with each recording identified only by a number.
