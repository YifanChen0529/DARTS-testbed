#!/usr/bin/env bash
LOG_DIR="${1:-sweep_logs_a1a2b1_061731}"
SWEEP_DIR="${2:-results_a1a2b1_20260521}"
EXPECTED=246

echo "════════ DARTS sweep status @ $(date +%H:%M:%S) ════════"
echo ""
echo "── Pane sentinels (done panes) ──"
if ls "$LOG_DIR/sentinels/" 2>/dev/null | grep -q .; then
  for f in "$LOG_DIR"/sentinels/PANE_DONE_*; do
    pane=$(basename "$f" | sed 's/PANE_DONE_//')
    rc=$(cat "$f")
    if [ "$rc" = "0" ]; then echo "  ✓ $pane (rc=0)"; else echo "  ✗ $pane (rc=$rc) ⚠"; fi
  done
else
  echo "  (none done yet)"
fi
echo ""
echo "── Aggregate progress (target=$EXPECTED) ──"
N_AGG=$(find "$SWEEP_DIR" -name "*_log.csv" 2>/dev/null | wc -l)
PCT=$(awk -v n=$N_AGG -v t=$EXPECTED 'BEGIN{printf "%.1f", n*100/t}')
echo "  $N_AGG / $EXPECTED cells aggregated ($PCT%)"
echo ""
echo "── Per-sweep breakdown ──"
for stress in light medium heavy energy_b04 energy_b03 energy_b02 medium_n2 medium_n6 medium_n8; do
  n=$(find "$SWEEP_DIR" -path "*/B1/$stress/*_log.csv" 2>/dev/null | wc -l)
  printf "  %-12s %d cells\n" "$stress" "$n"
done
echo ""
echo "── Newest 5 CSV updates ──"
find results -name "*_log.csv" -printf '%TY-%Tm-%Td %TH:%TM  %p\n' 2>/dev/null | sort | tail -5
echo ""
echo "── Monitor last 3 heartbeats ──"
tail -3 "$LOG_DIR/sweep_monitor.log" 2>/dev/null || echo "  (no monitor log yet)"
echo "═══════════════════════════════════════════════════════"