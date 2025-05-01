import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from itertools import product
import time
import argparse
import os
import random

def allocate(order_size, venues, lambda_over, lambda_under, theta_queue):
    """
    Allocate an order across venues to minimize cost.
    
    Implements the static Cont-Kukanov algorithm exactly as in the pseudocode.
    
    Parameters:
    - order_size: target shares to buy
    - venues: list of objects with .ask .ask_size .fee .rebate
    - lambda_over: cost penalty per extra share bought
    - lambda_under: cost penalty per unfilled share
    - theta_queue: queue-risk penalty (linear in total mis-execution)
    
    Returns:
    - best_split: list[int] shares sent to each venue
    - best_cost: float total expected cost of that split
    """
    step = globals().get("STEP_SIZE", 100)  # Search in configurable share chunks
    splits = [[]]  # Start with an empty allocation list
    
    for v in range(len(venues)):
        new_splits = []
        for alloc in splits:
            used = sum(alloc)
            max_v = min(order_size-used, venues[v]["ask_size"])
            for q in range(0, max_v+1, step):
                new_splits.append(alloc + [q])
        splits = new_splits

    best_cost = float('inf')
    best_split = []
    
    for alloc in splits:
        if sum(alloc) != order_size:
            continue
        cost = compute_cost(alloc, venues, order_size, lambda_over, lambda_under, theta_queue)
        if cost < best_cost:
            best_cost = cost
            best_split = alloc
    
    # If no valid allocation found, try to get as close as possible to the target
    if not best_split and splits:
        best_diff = float('inf')
        for alloc in splits:
            diff = abs(sum(alloc) - order_size)
            if diff < best_diff:
                best_diff = diff
                best_split = alloc
        
        # Only use this allocation if it's not empty
        if best_split:
            best_cost = compute_cost(best_split, venues, order_size, lambda_over, lambda_under, theta_queue)
    
    return best_split, best_cost

def compute_cost(split, venues, order_size, lambda_o, lambda_u, theta):
    """
    Compute the cost of a given order allocation.
    Exactly as defined in the pseudocode.
    """
    executed = 0
    cash_spent = 0.0
    
    for i in range(len(venues)):
        exe = min(split[i], venues[i]["ask_size"])
        executed += exe
        
        # Maker execution: price * executed * (1 - rebate%)
        cash_spent += exe * venues[i]["ask"] * (1.0 - venues[i]["rebate"])
        
        # Any shares you "tried" beyond displayed depth just roll forward,
        # but you don't double-credit them, so ignore here.
    
    underfill = max(order_size-executed, 0)
    overfill = max(executed-order_size, 0)
    risk_pen = theta * (underfill + overfill)
    cost_pen = lambda_u * underfill + lambda_o * overfill
    
    return cash_spent + risk_pen + cost_pen

def load_and_process_data(file_path, fee=0.0003, rebate=0.0002):
    """
    Process L1 data into snapshots for back-testing.
    
    Parameters:
    - file_path: Path to the CSV file
    - fee: Standard fee to apply to venues (default: 0.0003)
    - rebate: Standard rebate to apply to venues (default: 0.0002)
    
    Returns:
    - List of snapshots, each containing timestamp and venue data
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Load data
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows")
    
    # Verify required columns exist
    required_columns = ["ts_event", "publisher_id", "ask_px_00", "ask_sz_00"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in data file: {missing_columns}")
    
    # Extract unique timestamps
    unique_events = df["ts_event"].unique()
    print(f"Found {len(unique_events)} unique timestamps")
    
    # Create snapshots - one per timestamp
    snapshots = []
    
    for ts_event in unique_events:
        # For this timestamp, keep only the first message per publisher_id
        ts_df = df[df["ts_event"] == ts_event].drop_duplicates("publisher_id")
        
        # Build venue objects for this snapshot
        venues = []
        for _, row in ts_df.iterrows():
            venue = {
                "publisher_id": row["publisher_id"],
                "ask": row["ask_px_00"],
                "ask_size": row["ask_sz_00"],
                "fee": fee,  # Configurable fee
                "rebate": rebate  # Configurable rebate
            }
            venues.append(venue)
        
        # Add snapshot with timestamp and venues
        snapshots.append({
            "timestamp": ts_event,
            "venues": venues
        })
    
    return sorted(snapshots, key=lambda s: s["timestamp"])

def parameter_search(snapshots, order_size, search_config=None):
    """
    Search for optimal parameters for the Cont-Kukanov strategy.
    
    Parameters:
    - snapshots: List of market snapshots
    - order_size: Target order size to execute
    - search_config: Dictionary with parameter search configuration
                   If None, default ranges will be used
    
    Returns:
    - best_params: Dictionary with best parameters found
    - best_result: Result metrics for the best parameters
    - all_results: List of all parameter combinations and their results
    """
    # Default parameter ranges if not provided
    if search_config is None:
        search_config = {
            "lambda_over": [0.0001, 0.001, 0.002, 0.005, 0.01],
            "lambda_under": [0.001, 0.003, 0.005, 0.01, 0.02],
            "theta_queue": [0.0001, 0.0005, 0.001, 0.002, 0.005]
        }
    
    lambda_over_values = search_config["lambda_over"]
    lambda_under_values = search_config["lambda_under"]
    theta_queue_values = search_config["theta_queue"]
    
    best_params = None
    best_result = None
    best_avg_price = float('inf')
    
    # Track all results for reporting
    all_results = []
    
    # Use full dataset for fine search, subset for coarse search
    if len(lambda_over_values) > 5:  # Fine search has more points
        search_snapshots = snapshots
        print("Using full dataset for fine parameter search...")
    else:
        search_snapshots = snapshots[:len(snapshots)//2]
        print("Using subset of data for coarse parameter search...")
    
    print(f"Running parameter search with {len(search_snapshots)} snapshots...")
    start_time = time.time()
    
    total_combinations = len(lambda_over_values) * len(lambda_under_values) * len(theta_queue_values)
    count = 0
    
    for lo, lu, tq in product(lambda_over_values, lambda_under_values, theta_queue_values):
        count += 1
        if count % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Tested {count}/{total_combinations} parameter combinations. Elapsed: {elapsed:.2f}s")
        
        # Run backtest with these parameters
        result = run_backtest(search_snapshots, order_size, lo, lu, tq)
        avg_price = result["avg_price"]
        
        all_results.append({
            "lambda_over": lo,
            "lambda_under": lu,
            "theta_queue": tq,
            "avg_price": avg_price
        })
        
        # Update best parameters if this is better
        if avg_price < best_avg_price:
            best_avg_price = avg_price
            best_params = {"lambda_over": lo, "lambda_under": lu, "theta_queue": tq}
            best_result = result
    
    print(f"Parameter search complete. Best average price: {best_avg_price}")
    return best_params, best_result, all_results

def run_backtest(snapshots, order_size, lambda_over, lambda_under, theta_queue):
    """
    Run a backtest of the Cont-Kukanov strategy.
    """
    remaining = order_size
    total_filled = 0
    cash_spent = 0
    fills = []
    
    for snapshot in snapshots:
        if remaining <= 0:
            break
        
        venues = snapshot["venues"]
        if not venues:
            continue
        
        # Get allocation from the strategy
        allocation, _ = allocate(remaining, venues, lambda_over, lambda_under, theta_queue)
        
        if not allocation:
            continue
        
        # Execute the allocation
        snapshot_filled = 0
        snapshot_cost = 0
        
        for i, shares in enumerate(allocation):
            if i >= len(venues):
                break
                
            venue = venues[i]
            executed = min(shares, venue["ask_size"])
            
            if executed > 0:
                snapshot_filled += executed
                cost = executed * venue["ask"]
                snapshot_cost += cost
                
                fills.append({
                    "timestamp": snapshot["timestamp"],
                    "venue": venue["publisher_id"],
                    "shares": executed,
                    "price": venue["ask"],
                    "cost": cost
                })
        
        # Update remaining shares
        total_filled += snapshot_filled
        cash_spent += snapshot_cost
        remaining -= snapshot_filled
    
    # If still have remaining shares, assume market order at end
    if remaining > 0 and snapshots:
        last_snapshot = snapshots[-1]
        venues = last_snapshot["venues"]
        if venues:
            best_venue = min(venues, key=lambda v: v["ask"])
            
            cost = remaining * best_venue["ask"]
            cash_spent += cost
            total_filled += remaining
            
            fills.append({
                "timestamp": last_snapshot["timestamp"],
                "venue": best_venue["publisher_id"],
                "shares": remaining,
                "price": best_venue["ask"],
                "cost": cost
            })
    
    # Calculate average price
    avg_price = cash_spent / order_size if total_filled > 0 else 0
    
    return {
        "total_cash_spent": cash_spent,
        "avg_price": avg_price,
        "fills": fills
    }

def run_best_ask_baseline(snapshots, order_size, fee=0.0003):
    """
    Backtest the 'take the best ask' baseline.
    
    Parameters:
    - snapshots: List of market snapshots
    - order_size: Target order size to execute
    - fee: Taker fee as a percentage of trade price (default: 0.0003)
    """
    remaining = order_size
    total_filled = 0
    cash_spent = 0
    fills = []
    
    for snapshot in snapshots:
        if remaining <= 0:
            break
        
        venues = snapshot["venues"]
        if not venues:
            continue
        
        # Find venue with best ask
        best_venue = min(venues, key=lambda v: v["ask"])
        
        # Execute against best ask (as taker: pay fee%)
        executed = min(remaining, best_venue["ask_size"])
        cost = executed * best_venue["ask"] * (1.0 + fee)  # Pay taker fee as % of price
        
        if executed > 0:
            total_filled += executed
            cash_spent += cost
            remaining -= executed
            
            fills.append({
                "timestamp": snapshot["timestamp"],
                "venue": best_venue["publisher_id"],
                "shares": executed,
                "price": best_venue["ask"],
                "cost": cost
            })
    
    # If still have remaining shares, assume market order at end
    if remaining > 0 and snapshots:
        last_snapshot = snapshots[-1]
        venues = last_snapshot["venues"]
        if venues:
            best_venue = min(venues, key=lambda v: v["ask"])
            
            cost = remaining * best_venue["ask"] * (1.0 + fee)  # Pay taker fee as % of price
            cash_spent += cost
            total_filled += remaining
            
            fills.append({
                "timestamp": last_snapshot["timestamp"],
                "venue": best_venue["publisher_id"],
                "shares": remaining,
                "price": best_venue["ask"],
                "cost": cost
            })
    
    # Calculate average price
    avg_price = cash_spent / order_size if total_filled > 0 else 0
    
    return {
        "total_cash_spent": cash_spent,
        "avg_price": avg_price,
        "fills": fills
    }

def run_twap_baseline(snapshots, order_size, bucket_seconds=60, fee=0.0003):
    """
    Backtest the TWAP baseline with 60-second buckets.
    
    Parameters:
    - snapshots: List of market snapshots
    - order_size: Target order size to execute
    - bucket_seconds: Size of TWAP time buckets in seconds (default: 60)
    - fee: Taker fee as a percentage of trade price (default: 0.0003)
    """
    if not snapshots:
        return {
            "total_cash_spent": 0,
            "avg_price": 0,
            "fills": []
        }
    
    # Extract timestamps as datetime objects
    for snapshot in snapshots:
        snapshot["dt"] = pd.to_datetime(snapshot["timestamp"])
    
    # Find min and max timestamps
    min_dt = min(snapshot["dt"] for snapshot in snapshots)
    max_dt = max(snapshot["dt"] for snapshot in snapshots)
    
    # Create time buckets
    total_seconds = (max_dt - min_dt).total_seconds()
    num_buckets = max(1, int(total_seconds / bucket_seconds))
    
    # Assign snapshots to buckets
    buckets = {}
    for snapshot in snapshots:
        bucket_idx = int((snapshot["dt"] - min_dt).total_seconds() / bucket_seconds)
        bucket_idx = min(bucket_idx, num_buckets - 1)  # Ensure within bounds
        
        if bucket_idx not in buckets:
            buckets[bucket_idx] = []
        buckets[bucket_idx].append(snapshot)
    
    # Calculate shares per bucket
    shares_per_bucket = order_size / num_buckets
    
    # Execute TWAP
    remaining = order_size
    total_filled = 0
    cash_spent = 0
    fills = []
    
    for bucket_idx in range(num_buckets):
        if remaining <= 0:
            break
        
        # Get snapshots for this bucket
        bucket_snapshots = buckets.get(bucket_idx, [])
        if not bucket_snapshots:
            continue
        
        # Use first snapshot in bucket
        snapshot = bucket_snapshots[0]
        venues = snapshot["venues"]
        if not venues:
            continue
        
        # Find venue with best ask
        best_venue = min(venues, key=lambda v: v["ask"])
        
        # Execute bucket's share (as taker: pay fee%)
        executed = min(shares_per_bucket, remaining, best_venue["ask_size"])
        cost = executed * best_venue["ask"] * (1.0 + fee)  # Pay taker fee as % of price
        
        if executed > 0:
            total_filled += executed
            cash_spent += cost
            remaining -= executed
            
            fills.append({
                "timestamp": snapshot["timestamp"],
                "venue": best_venue["publisher_id"],
                "shares": executed,
                "price": best_venue["ask"],
                "cost": cost
            })
    
    # If still have remaining shares, assume market order at end
    if remaining > 0 and snapshots:
        last_snapshot = snapshots[-1]
        venues = last_snapshot["venues"]
        if venues:
            best_venue = min(venues, key=lambda v: v["ask"])
            
            cost = remaining * best_venue["ask"] * (1.0 + fee)  # Pay taker fee as % of price
            cash_spent += cost
            total_filled += remaining
            
            fills.append({
                "timestamp": last_snapshot["timestamp"],
                "venue": best_venue["publisher_id"],
                "shares": remaining,
                "price": best_venue["ask"],
                "cost": cost
            })
    
    # Calculate average price
    avg_price = cash_spent / order_size if total_filled > 0 else 0
    
    return {
        "total_cash_spent": cash_spent,
        "avg_price": avg_price,
        "fills": fills
    }

def run_vwap_baseline(snapshots, order_size, fee=0.0003):
    """
    Backtest the VWAP baseline.
    
    Parameters:
    - snapshots: List of market snapshots
    - order_size: Target order size to execute
    - fee: Taker fee as a percentage of trade price (default: 0.0003)
    """
    if not snapshots:
        return {
            "total_cash_spent": 0,
            "avg_price": 0,
            "fills": []
        }
    
    # Calculate total volume and VWAP
    total_volume = 0
    total_value = 0
    for snapshot in snapshots:
        venues = snapshot["venues"]
        if not venues:
            continue
            
        # Find venue with best ask
        best_venue = min(venues, key=lambda v: v["ask"])
        volume = best_venue["ask_size"]
        value = volume * best_venue["ask"]
        
        total_volume += volume
        total_value += value
    
    if total_volume == 0:
        return {
            "total_cash_spent": 0,
            "avg_price": 0,
            "fills": []
        }
    
    vwap = total_value / total_volume
    
    # Execute VWAP
    remaining = order_size
    total_filled = 0
    cash_spent = 0
    fills = []
    
    for snapshot in snapshots:
        if remaining <= 0:
            break
            
        venues = snapshot["venues"]
        if not venues:
            continue
            
        # Find venue with best ask
        best_venue = min(venues, key=lambda v: v["ask"])
        
        # Calculate target volume for this snapshot
        snapshot_volume = best_venue["ask_size"]
        target_volume = (snapshot_volume / total_volume) * order_size
        
        # Execute target volume (as taker: pay fee%)
        executed = min(target_volume, remaining, best_venue["ask_size"])
        cost = executed * best_venue["ask"] * (1.0 + fee)  # Pay taker fee as % of price
        
        if executed > 0:
            total_filled += executed
            cash_spent += cost
            remaining -= executed
            
            fills.append({
                "timestamp": snapshot["timestamp"],
                "venue": best_venue["publisher_id"],
                "shares": executed,
                "price": best_venue["ask"],
                "cost": cost
            })
    
    # If still have remaining shares, assume market order at end
    if remaining > 0 and snapshots:
        last_snapshot = snapshots[-1]
        venues = last_snapshot["venues"]
        if venues:
            best_venue = min(venues, key=lambda v: v["ask"])
            
            cost = remaining * best_venue["ask"] * (1.0 + fee)  # Pay taker fee as % of price
            cash_spent += cost
            total_filled += remaining
            
            fills.append({
                "timestamp": last_snapshot["timestamp"],
                "venue": best_venue["publisher_id"],
                "shares": remaining,
                "price": best_venue["ask"],
                "cost": cost
            })
    
    # Calculate average price
    avg_price = cash_spent / order_size if total_filled > 0 else 0
    
    return {
        "total_cash_spent": cash_spent,
        "avg_price": avg_price,
        "fills": fills
    }

def plot_cumulative_costs(best_fills, best_ask_fills, twap_fills, vwap_fills, filename="results.pdf"):
    """
    Create a plot of the cumulative costs for all strategies.
    """
    plt.figure(figsize=(12, 6))
    
    # Convert to DataFrames for easier manipulation
    best_df = pd.DataFrame(best_fills)
    best_ask_df = pd.DataFrame(best_ask_fills)
    twap_df = pd.DataFrame(twap_fills)
    vwap_df = pd.DataFrame(vwap_fills)
    
    # Sort by timestamp
    if not best_df.empty:
        best_df['timestamp'] = pd.to_datetime(best_df['timestamp'])
        best_df = best_df.sort_values('timestamp')
    
    if not best_ask_df.empty:
        best_ask_df['timestamp'] = pd.to_datetime(best_ask_df['timestamp'])
        best_ask_df = best_ask_df.sort_values('timestamp')
    
    if not twap_df.empty:
        twap_df['timestamp'] = pd.to_datetime(twap_df['timestamp'])
        twap_df = twap_df.sort_values('timestamp')
    
    if not vwap_df.empty:
        vwap_df['timestamp'] = pd.to_datetime(vwap_df['timestamp'])
        vwap_df = vwap_df.sort_values('timestamp')
    
    # Calculate cumulative costs
    if not best_df.empty:
        best_df['cumulative_cost'] = best_df['cost'].cumsum()
        plt.plot(best_df['timestamp'], best_df['cumulative_cost'], 'b-', label='Cont-Kukanov (Tuned)')
    
    if not best_ask_df.empty:
        best_ask_df['cumulative_cost'] = best_ask_df['cost'].cumsum()
        plt.plot(best_ask_df['timestamp'], best_ask_df['cumulative_cost'], 'r--', label='Best Ask')
    
    if not twap_df.empty:
        twap_df['cumulative_cost'] = twap_df['cost'].cumsum()
        plt.plot(twap_df['timestamp'], twap_df['cumulative_cost'], 'g-.', label='TWAP')
    
    if not vwap_df.empty:
        vwap_df['cumulative_cost'] = vwap_df['cost'].cumsum()
        plt.plot(vwap_df['timestamp'], vwap_df['cumulative_cost'], 'm:', label='VWAP')
    
    plt.xlabel('Time')
    plt.ylabel('Cumulative Cost ($)')
    plt.title('Cumulative Execution Costs')
    plt.legend()
    plt.grid(True)
    
    # Format x-axis
    plt.gcf().autofmt_xdate()
    
    # Save figure
    plt.savefig(filename)
    plt.close()
    
    print(f"Cumulative cost plot saved to {filename}")

def calculate_savings_bps(strategy_price, baseline_price):
    """
    Calculate savings in basis points relative to a baseline.
    """
    if baseline_price <= 0:
        return 0
    return (baseline_price - strategy_price) / baseline_price * 10000  # 10000 converts to basis points

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Cont & Kukanov Order Router Backtest')
    
    parser.add_argument('--data-file', type=str, default='l1_day.csv',
                        help='Path to the L1 data CSV file (default: l1_day.csv)')
    
    parser.add_argument('--order-size', type=int, default=5000,
                        help='Target order size in shares (default: 5000)')
    
    parser.add_argument('--fee', type=float, default=0.0003,
                        help='Standard fee per share (default: 0.0003)')
    
    parser.add_argument('--rebate', type=float, default=0.0002,
                        help='Standard rebate per share (default: 0.0002)')
    
    parser.add_argument('--plot-file', type=str, default='results.pdf',
                        help='File to save the cumulative cost plot (default: results.pdf)')
    
    parser.add_argument('--twap-bucket', type=int, default=60,
                        help='TWAP bucket size in seconds (default: 60)')
    
    parser.add_argument('--step-size', type=int, default=1,
                        help='Chunk size for allocation search (default: 1 share)')
    
    return parser.parse_args()

def main():
    """
    Main function to run the backtest.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Set global step size for allocation
    globals()["STEP_SIZE"] = args.step_size
    
    # Load and process data
    snapshots = load_and_process_data(args.data_file, args.fee, args.rebate)
    order_size = args.order_size
    
    # Phase 1: Wide coarse grid on half the data
    print("Phase 1: Running wide coarse parameter search...")
    coarse_config = {
        "lambda_over": np.linspace(1e-5, 1e-3, 10).tolist(),
        "lambda_under": np.linspace(1e-4, 1e-2, 10).tolist(),
        "theta_queue": np.linspace(1e-5, 1e-3, 10).tolist()
    }
    
    # Search for optimal parameters on a subset of data
    best_params, _, _ = parameter_search(snapshots, order_size, coarse_config)
    
    # Phase 2: Dense local grid on full data
    print("\nPhase 2: Running dense local parameter search...")
    lo0, lu0, tq0 = best_params["lambda_over"], best_params["lambda_under"], best_params["theta_queue"]
    dense_config = {
        "lambda_over": [lo0 * f for f in np.linspace(0.5, 1.5, 15)],
        "lambda_under": [lu0 * f for f in np.linspace(0.5, 1.5, 15)],
        "theta_queue": [tq0 * f for f in np.linspace(0.5, 1.5, 15)]
    }
    
    # Run dense search on full dataset
    best_params, best_result, _ = parameter_search(snapshots, order_size, dense_config)
    
    # Phase 3: Stochastic hill-climbing
    print("\nPhase 3: Running stochastic hill-climbing...")
    best_price = best_result["avg_price"]
    for i in range(500):
        if i % 50 == 0:
            print(f"Stochastic search iteration {i}/500...")
        
        # Generate random parameters within ±10% of best found
        lo = lo0 * random.uniform(0.9, 1.1)
        lu = lu0 * random.uniform(0.9, 1.1)
        tq = tq0 * random.uniform(0.9, 1.1)
        
        # Run backtest with these parameters
        res = run_backtest(snapshots, order_size, lo, lu, tq)
        
        # Update if we found a better solution
        if res["avg_price"] < best_price:
            best_price = res["avg_price"]
            best_params = {"lambda_over": lo, "lambda_under": lu, "theta_queue": tq}
            best_result = res
            print(f"Found better parameters: {best_params} with price {best_price}")
    
    # Run baselines
    print("\nRunning baseline strategies...")
    best_ask_result = run_best_ask_baseline(snapshots, order_size, args.fee)
    twap_result = run_twap_baseline(snapshots, order_size, args.twap_bucket, args.fee)
    vwap_result = run_vwap_baseline(snapshots, order_size, args.fee)
    
    # Calculate savings in basis points
    savings_vs_best_ask = calculate_savings_bps(best_result["avg_price"], best_ask_result["avg_price"])
    savings_vs_twap = calculate_savings_bps(best_result["avg_price"], twap_result["avg_price"])
    savings_vs_vwap = calculate_savings_bps(best_result["avg_price"], vwap_result["avg_price"])
    
    # Create output JSON
    output = {
        "best_parameters": {
            "lambda_over": best_params["lambda_over"],
            "lambda_under": best_params["lambda_under"],
            "theta_queue": best_params["theta_queue"]
        },
        "cont_kukanov_performance": {
            "total_cash_spent": best_result["total_cash_spent"],
            "avg_price": best_result["avg_price"]
        },
        "best_ask_performance": {
            "total_cash_spent": best_ask_result["total_cash_spent"],
            "avg_price": best_ask_result["avg_price"]
        },
        "twap_performance": {
            "total_cash_spent": twap_result["total_cash_spent"],
            "avg_price": twap_result["avg_price"]
        },
        "vwap_performance": {
            "total_cash_spent": vwap_result["total_cash_spent"],
            "avg_price": vwap_result["avg_price"]
        },
        "savings_bps": {
            "vs_best_ask": savings_vs_best_ask,
            "vs_twap": savings_vs_twap,
            "vs_vwap": savings_vs_vwap
        }
    }
    
    # Plot cumulative costs
    plot_cumulative_costs(
        best_result["fills"],
        best_ask_result["fills"],
        twap_result["fills"],
        vwap_result["fills"],
        args.plot_file
    )
    
    # Print JSON output
    print(json.dumps(output, indent=2))
    
if __name__ == "__main__":
    main()