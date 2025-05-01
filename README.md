
# Cont & Kukanov SOR Backtest

This repository implements and tunes the static Cont & Kukanov Smart Order Router. It splits a 5 000-share buy order across multiple venues to minimize execution cost, then benchmarks against three baselines (Best Ask, TWAP, VWAP) over a nine-minute L1 data window.

---

## Files to Push

- `backtest.py` — standalone script (only `numpy`, `pandas`, `matplotlib` + stdlib)
- `README.md` — this file
- `results.pdf` — cumulative-cost plot

---

## Usage

1. **Prepare environment**  
   ```bash
   python -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate         # Windows
   pip install pandas numpy matplotlib
   ```

2. **Run backtest**  
   ```bash
   python backtest.py
   ```
   Outputs JSON metrics and saves `results.pdf`.

3. **Options**  
   ```bash
   python backtest.py \
     --data-file l1_day.csv \
     --order-size 5000 \
     --fee 0.0003 \
     --rebate 0.0002 \
     --plot-file results.pdf \
     --twap-bucket 60 \
     --step-size 1
   ```

---

## Code Structure

- **`allocate()`**  
  Implements the static Cont & Kukanov split (pseudocode exact).  
- **`compute_cost()`**  
  Cash + rebate + over/underfill + queue-risk penalties.  
- **Data loader**  
  Reads `l1_day.csv`, deduplicates per `ts_event` & `publisher_id`, builds snapshots.
- **`run_backtest()`**  
  Feeds snapshots into the allocator, rolls unfilled shares forward, final market fill.
- **Parameter search**  
  1. **Coarse grid** (10×10×10) on half the data  
  2. **Dense grid** (15×15×15) around best coarse result on full data  
  3. **Stochastic hill-climbing** (±10%) for 500 iterations  
- **Baselines**  
  - **Best Ask** (taker fee 0.0003)  
  - **TWAP** (60 s buckets, taker fee 0.0003)  
  - **VWAP** (weight by displayed size, taker fee 0.0003)  
- **Plot**  
  Cumulative cost comparison saved in `results.pdf`.

---

## Tuned Parameters & Results

```json
{
  "best_parameters": {
    "lambda_over": 5e-06,
    "lambda_under": 5e-05,
    "theta_queue": 5e-06
  },
  "cont_kukanov_performance": {
    "avg_price": 222.82045600000006
  },
  "best_ask_performance": {
    "avg_price": 222.88730213680006
  },
  "twap_performance": {
    "avg_price": 223.31371002080002
  },
  "vwap_performance": {
    "avg_price": 223.13075095913368
  },
  "savings_bps": {
    "vs_best_ask": 3.00,
    "vs_twap": 22.09,
    "vs_vwap": 13.91
  }
}
```

- **Cont-Kukanov** vs **Best Ask**: +3 bps  
- vs **TWAP**: +22 bps  
- vs **VWAP**: +14 bps  

---

## Suggested Improvement

Current backtest assumes immediate fills up to displayed size.  
**Enhance realism by modeling queue position & fill probability**:
1. Estimate your limit order’s position in each book.  
2. Use a probabilistic fill model based on queue depth, order‐flow imbalance, and cancellation rates.  
3. Simulate partial fills, update queue states, and recompute allocation dynamically.

This captures real-world latency and improves the fidelity of execution cost estimates.  
