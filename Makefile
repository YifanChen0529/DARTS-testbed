# DARTS-Simulation — RTSS'26 Paper #117 artifact regeneration
#
# Curated data roots (populated by AE task T7); override via make DATA=... etc.
PY      ?= python3
DATA    ?= data
OUT     ?= out
BEH_A1  ?= $(DATA)/raw/behavioral/sweep_a1a2b1
BEH_LEG ?= $(DATA)/raw/behavioral/legacy_medium
BEH_D1  ?= $(DATA)/raw/behavioral/sweep_d1

.DELETE_ON_ERROR:

# Order-only directory for redirected MD / PDF / CSV outputs.
$(OUT):
	mkdir -p $@

# ---------------------------------------------------------------------------
# Real FILE targets — stdout MD intermediates (published redirect pattern)
# ---------------------------------------------------------------------------

# Published parameterization verified from committed vg_t3_combined.md header
# (δ_auc=30.0, H=1000.0, J̄=0.0, u_i,max=0.22, h=50.0, start_pos=(0,0)) and
# scripts/vg_t3_combined.py docstring / defaults; values encoded explicitly.
$(OUT)/vg_t3_combined.md: | $(OUT)
	$(PY) scripts/vg_t3_combined.py \
		--new_sweep_dir $(BEH_A1) \
		--old_results_dir $(BEH_LEG) \
		--delta_auc_ms 30 \
		--H_ms 1000 \
		--j_bar_ms 0 \
		--ui_max_mps 0.22 \
		--h_comm_ms 50 \
		--start_x 0 \
		--start_y 0 \
		> $@

$(OUT)/vg_analysis_output.md: | $(OUT)
	$(PY) scripts/vg_analysis.py \
		--new_sweep_dir $(BEH_A1) \
		--old_results_dir $(BEH_LEG) \
		> $@

$(OUT)/recovered_n3_ffl.md: | $(OUT)
	$(PY) scripts/recover_63col.py \
		--new_sweep_dir $(BEH_A1) \
		> $@

# ---------------------------------------------------------------------------
# Phony AE element targets (Sec. 3 inventory)
# ---------------------------------------------------------------------------

.PHONY: all demo test test-all clean verify \
	fig3 fig4 fig5 fig6 tab2 tab3 tab4 tab5

tab2: $(OUT)/vg_t3_combined.md

tab3: | $(OUT)
	$(PY) scripts/summarize_sweep.py \
		--results_dir $(BEH_D1) \
		--out_dir $(OUT)

tab4: $(OUT)/vg_analysis_output.md

tab5: $(OUT)/recovered_n3_ffl.md

fig3: $(OUT)/vg_t3_combined.md | $(OUT)
	$(PY) scripts/fig0_headline_bar.py \
		--new_sweep_dir $(BEH_A1) \
		--out_dir $(OUT)
	$(PY) scripts/fig3_t3_bound_tightness.py \
		--input_md $(OUT)/vg_t3_combined.md \
		--out_dir $(OUT)

fig4: | $(OUT)
	$(PY) scripts/fig1_b1_grid.py \
		--new_sweep_dir $(BEH_A1) \
		--out_dir $(OUT)

# Timing-panel fallback constants are the frozen published values (May 25 freeze
# decision, documented in DATA.md). Default regeneration uses CSV-derived values
# where the curated tree yields them and the frozen constants elsewhere; pass
# --strict manually to force CSV-only derivation.
fig5: | $(OUT)
	$(PY) scripts/fig2_generalization_scaling.py \
		--new_sweep_dir $(BEH_A1) \
		--old_results_dir $(BEH_LEG) \
		--out_dir $(OUT)

fig6: $(OUT)/recovered_n3_ffl.md | $(OUT)
	$(PY) scripts/fig4_d1_heatmap.py \
		--input_md $(OUT)/recovered_n3_ffl.md \
		--out_dir $(OUT)

verify:
	$(PY) scripts/ae_spotcheck_tab3.py \
		--results_dir $(BEH_D1) \
		--root_label curated

all: fig3 fig4 fig5 fig6 tab2 tab3 tab4 tab5

demo:
	$(PY) run_artifact_demo.py --benchmark B1 --stress medium --episodes 200 --seed 0

test:
	$(PY) -m pytest -q -m "not slow" tests/

test-all:
	$(PY) -m pytest -q tests/

# Remove $(OUT) contents only (keep the directory if present).
clean:
	@if [ -d "$(OUT)" ]; then find "$(OUT)" -mindepth 1 -delete; fi
