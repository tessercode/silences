# Playing the Silences: reproduction targets (see REPRODUCE.md for what each checks and how long it takes).
#
#   make setup          a Python 3.12 virtual environment with the pinned requirements and the NLTK corpora
#   make paper          tier 1: the records -> every table, figure and number in the paper (seconds)
#   make check          fast checks: unit tests, every firmware golden, every generated header
#   make evaluate       tier 2: the shipped models -> the superseded 5,000-transmission records, then compared (about 10 min)
#   make evaluate-matched  tier 2: every model of the grid -> the matched-data record (records/matched/), then compared (about 40 min)
#   make evaluate-spike-fed  tier 2: the spike-fed families -> the record of the paper (records/spike_fed/), then compared (about 25 min; needs the matched tables)
#   make spike-character   the spike statistics and timing-destroying controls of both LIF forms (records/spike_fed/spikes_*.json, about 40 min)
#   make train-spk40 SEED=0 CORPUS=op12
#                       tier 3: retrain one seed of the network of record (the LIF fed spikes) and check it is bit-identical
#   make train-lif SEED=0 / train-mlp SEED=0 [CORPUS=op12|gen]; train-spk80 / train-lif80 SEED=0 CORPUS=op12
#                       tier 3: retrain one seed and check it is bit-identical to the shipped model
#                       (CORPUS=op12 is the matched corpus; without CORPUS, the original 5,000-transmission corpus)
#   make train-bigru [ARM=rep|fist|shared]
#                       tier 3: retrain the BiGRU pool, or one of its retrained arms (CUDA GPU, about 5 min each)
#   make lm             rebuild the character language model and check it is byte-identical
#   make firmware-host  every board's host gate (gcc, no SDK)
#   make firmware-renode   every board on the emulated RP2040 (Pico SDK and Renode, see firmware/README.md)

PYTHON ?= .venv/bin/python
SEED ?= 0
export PYTHONPATH := $(CURDIR)

.PHONY: setup paper check evaluate evaluate-matched evaluate-spike-fed spike-character train-lif train-mlp train-spk40 train-spk80 train-lif80 train-bigru lm collision firmware-host firmware-renode
CORPUS ?=
ARM ?=

setup:
	python3.12 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	.venv/bin/python -m nltk.downloader words brown

paper:
	$(PYTHON) paper/make_figures.py
	git diff --exit-code --stat -- paper/
	@echo "tier 1: every number, table and figure is identical to the committed ones"

check:
	$(PYTHON) -m pytest -q tests
	$(PYTHON) -m silences.golden.check
	$(PYTHON) -m silences.golden.headers

# The records are rewritten in place; git diff and tools/compare_records.py show that nothing but provenance changed.
evaluate:
	$(PYTHON) -m silences.evaluation.tables
	$(PYTHON) -m silences.evaluation.random_letters
	$(PYTHON) -m silences.evaluation.decoder_weights
	$(PYTHON) tools/compare_records.py --git records/random_letters records/decoder_weights

evaluate-matched:
	$(PYTHON) -m silences.evaluation.matched
	$(PYTHON) tools/compare_records.py --git records/matched

evaluate-spike-fed:
	$(PYTHON) -m silences.evaluation.spike_fed
	$(PYTHON) tools/compare_records.py --git records/spike_fed

spike-character:
	$(PYTHON) -m silences.evaluation.spike_character spk40
	$(PYTHON) -m silences.evaluation.spike_character lif
	$(PYTHON) tools/compare_records.py --git records/spike_fed

train-lif train-mlp train-spk40 train-spk80 train-lif80:
	$(PYTHON) -m silences.training.window_nets $(@:train-%=%) $(SEED) $(if $(CORPUS),--corpus $(CORPUS),)

train-bigru:
	$(PYTHON) -m silences.training.bigru $(if $(ARM),--arm $(ARM),)

lm:
	$(PYTHON) -m silences.decoder.charlm build

collision:
	$(PYTHON) -m silences.collision
	git diff --exit-code --stat -- records/collision.json

firmware-host:
	PY=$(abspath $(PYTHON)) firmware/check_host.sh

firmware-renode:
	PYTHON=$(abspath $(PYTHON)) firmware/check_renode.sh
