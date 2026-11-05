import pandas as pd

def compute_throughput(df, time_points):
    # window length in seconds (from config throughput_window, e.g. 10min -> 600)
    window_sec = (time_points[1] - time_points[0]).total_seconds()
    records = []
    for i in range(len(time_points) - 1):
        # window start
        t_start = time_points[i]
        # window ends = start time + window length
        t_end = time_points[i + 1]
        # Determine active transfer count: the transfer started at or before t_start, and ends after t_start
        # at the exact time, concurrent transfers
        time_mask = (df['start'] < t_end) & (df['complete'] > t_start)
        active = df[time_mask]
        # If no transfers are active in this window, skip it
        if len(active) == 0:
            records.append(0.0)
            continue
        # Compute the throughput within this window
        overlap_start = active['start'].clip(lower=t_start)
        overlap_end = active['complete'].clip(upper=t_end)
        # How long this transfer is actually active within the window
        overlap_sec = (overlap_end - overlap_start).dt.total_seconds()
        total_sec = (active['complete'] - active['start']).dt.total_seconds()
        bytes_in_window = active['volume'] * (overlap_sec / total_sec)
        #Assuming a uniform transfer rate, allocate the total volume to this window (and unit calculation)
        records.append(bytes_in_window.sum() / window_sec)  # bytes/s
    return records


def compute_active(df, time_points):
    counts = []
    for i in range(len(time_points) - 1):
        t_start = time_points[i]
        t_end = time_points[i + 1]
        # count every transfer that exists at any moment within [t_start, t_end)
        time_mask = (df['start'] < t_end) & (df['complete'] > t_start)
        counts.append(int(time_mask.sum()))
    return counts

def build_model_data(df, sources, cfg):
    tp_win, agg_win = cfg["throughput_window"], cfg["agg_window"]

    # ddavila: the next 2 blocks creates the set of time windows to be used to calculate
    # active transfers (1hr) and throughput (1m) Change in default.yaml
    # Get time axis, make the start time floored to the minute, and end time celled to minute.

    time_points = pd.date_range(df["start"].min().floor(tp_win),
                                df["complete"].max().ceil(tp_win), freq=tp_win)

    # hourly time axis for active transfer count
    agg_points = pd.date_range(df["start"].min().floor(agg_win),
                               df["complete"].max().ceil(agg_win), freq=agg_win)

    # ddavila: Add comments: calls the functions that calculate active transfers and throughput per site
    tp_cols, active_cols = {}, {}
    for bucket, g in df.groupby("source", observed=True):
        tp_cols[bucket] = compute_throughput(g, time_points)
        active_cols[bucket] = compute_active(g, agg_points)

    # ddavila: Add comments: creates a new DataFrame based on the dictionaries created above
    out = pd.concat(
        {
            "active_transfers": pd.DataFrame(active_cols, index=agg_points[:-1]),
            "throughput": pd.DataFrame(tp_cols, index=time_points[:-1]).resample(agg_win).mean(),
        },
        axis=1,
    ).fillna(0)

    # ddavila: Add comments: The next few cells just change the name and order of the columns (formating)
    out.columns = [f"{metric}_{bucket}" for metric, bucket in out.columns]

    # order of sites based on it's throughput
    buckets = [s for s in sources if s in active_cols]
    ordered = [c for b in buckets for c in (f"active_transfers_{b}", f"throughput_{b}")]
    out = out[ordered]
    active = [c for c in ordered if c.startswith("active_transfers_")]
    out[active] = out[active].astype(int)

    # Add the total throughput column
    total = pd.Series(compute_throughput(df, time_points), index=time_points[:-1]).resample(agg_win).mean()
    out = out.join(total.rename("total_throughput"))
    return out, buckets