import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib

# Force background rendering for speed
matplotlib.use('Agg')

# ---------------------------------------------------------
# 1. Global Configurations
# ---------------------------------------------------------
cap_sizes = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 
             5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]

# for original ADR data
thresh_map = {
    10.0: 1.9, 9.5: 1.9, 9.0: 1.9, 8.5: 1.9, 8.0: 1.9, 
    7.5: 1.9, 7.0: 1.9, 6.5: 1.9, 6.0: 1.9, 5.5: 1.9, 
    5.0: 1.9, 4.5: 1.9, 4.0: 1.9, 3.5: 1.9, 3.0: 1.95, 
    2.5: 1.95, 2.0: 2.0, 1.5: 2.05, 1.0: 2.2, 0.5: 2.65
}

payloadOption = "daily_random_20_255"
trainingPayload = "daily_random_20_255"
trainingCapSizeStr = "random_0.5_10"
tsf_max = 1000
training_neg_inaction_reward = -0.5
gamma = 0.99
pos_reward = 1.0
off_reward = 6.0
steps_per_day = 2880

# Labels and Styling
labels_all = ['Agent (ITI)', 'Agent (Off-Time)', 'Opt. Static Thresh.', 'Static (1.9V)', 'Static (3.45V)', 'Approx. Pred.', 'ST Oracle', 'AsTAR']
agent_labels = ['Agent (ITI)', 'Agent (Off-Time)']
baseline_labels = ['Opt. Static Thresh.', 'Static (1.9V)', 'Static (3.45V)', 'Approx. Pred.', 'ST Oracle', 'AsTAR']

colors_dict = {
    'Agent (ITI)': '#000000',
    'Agent (Off-Time)': '#E69F00',
    'Opt. Static Thresh.': '#56B4E9',
    'Static (1.9V)': '#009E73',
    'Static (3.45V)': '#F0E442',
    'Approx. Pred.': '#0072B2',
    'ST Oracle': '#D55E00',
    'AsTAR': '#CC79A7'
}

markers_dict = {
    'Agent (ITI)': 'o',
    'Agent (Off-Time)': 's',
    'Opt. Static Thresh.': '^',
    'Static (1.9V)': 'v',
    'Static (3.45V)': '<',
    'Approx. Pred.': 'D',
    'ST Oracle': 'p',
    'AsTAR': 'X'
}

linestyles_dict = {
    'Agent (ITI)': '-',
    'Agent (Off-Time)': '-',
    'Opt. Static Thresh.': '-',
    'Static (1.9V)': '--',
    'Static (3.45V)': '--',
    'Approx. Pred.': '-',
    'ST Oracle': '-',
    'AsTAR': '-'
}

seeds = [42, 123, 456, 789, 999, 0, 222, 333, 444, 555, 10, 20, 30, 40, 50]
# ADDED 'total_off_time' TO METRICS
metric_keys = ['mean_iti', 'mean_daily_success', 'median_survival', 'median_max_daily_iti', 'median_off_time_duration', 'total_off_time']

# Data structures:
# For RL agents: store trajectories per seed: trend_data_seeds[metric][agent_label][seed] = [val_for_c1, val_for_c2, ...]
trend_data_seeds = {
    metric: {
        label: {s: [] for s in seeds} for label in agent_labels
    } for metric in metric_keys
}

# For static baselines: store single curve across sizes
trend_data_baselines = {
    metric: {
        label: [] for label in baseline_labels
    } for metric in metric_keys
}

# ---------------------------------------------------------
# 2. Helper Functions
# ---------------------------------------------------------
def load_agent_data(path):
    if not os.path.exists(path):
        return np.array([])
    return pd.read_csv(path, header=None).iloc[:, 0].to_numpy()

def get_tbs(success_array, agent_type=None, eps=0.01):
    if agent_type == "jitter":
        idx = np.where((success_array > pos_reward - eps) & (success_array < 6.0))[0]
    elif agent_type == "off_time":
        idx = np.where((success_array > 0.0) & (success_array < pos_reward + eps))[0]
    else:
        idx = np.where(success_array == 1)[0]
    return np.diff(idx) if len(idx) >= 2 else np.array([])

def get_daily_success_counts(success_array, agent_type=None, eps=0.01):
    if agent_type == "jitter":
        binary_success = ((success_array > pos_reward - eps) & (success_array < 6.0)).astype(int)
    elif agent_type == "off_time":
        binary_success = ((success_array > 0.0) & (success_array < pos_reward + eps)).astype(int)
    else:
        binary_success = np.copy(success_array)
    
    num_days = len(binary_success) // steps_per_day
    if num_days == 0: return np.array([]) 
    truncated = binary_success[:num_days * steps_per_day]
    return np.sum(truncated.reshape((num_days, steps_per_day)), axis=1)

def get_daily_max_iti(success_array, agent_type=None, eps=0.01):
    if agent_type == "jitter":
        binary_success = ((success_array > pos_reward - eps) & (success_array < 6.0)).astype(int)
    elif agent_type == "off_time":
        binary_success = ((success_array > 0.0) & (success_array < pos_reward + eps)).astype(int)
    else:
        binary_success = np.copy(success_array)
    
    local_num_days = len(binary_success) // steps_per_day
    if local_num_days == 0: return np.array([]) 
    
    truncated = binary_success[:local_num_days * steps_per_day]
    daily_chunks = truncated.reshape((local_num_days, steps_per_day))
    
    daily_max_itis = []
    for day_chunk in daily_chunks:
        success_indices = np.where(day_chunk == 1)[0]
        if len(success_indices) == 0:
            daily_max_itis.append(steps_per_day)
        elif len(success_indices) == 1:
            max_gap = max(success_indices[0], steps_per_day - success_indices[0])
            daily_max_itis.append(max_gap)
        else:
            gaps = np.diff(success_indices)
            first_gap = success_indices[0] 
            last_gap = steps_per_day - success_indices[-1] 
            daily_max_itis.append(max(np.max(gaps), first_gap, last_gap))
    return np.array(daily_max_itis)

def get_survival_time(state_array, agent_type=None, off_r=6.0, eps=0.01):
    if agent_type in ["jitter", "off_time"]:
        is_on = ~((state_array > off_r - eps) & (state_array < off_r + eps))
    else:
        is_on = np.array(state_array).astype(bool)
        
    padded = np.pad(is_on.astype(int), (1, 1), mode='constant', constant_values=0)
    diffs = np.diff(padded)
    return np.where(diffs == -1)[0] - np.where(diffs == 1)[0]

def get_off_time_durations(state_array, agent_type=None, off_r=6.0, eps=0.01):
    if agent_type in ["jitter", "off_time"]:
        is_off = (state_array > off_r - eps) & (state_array < off_r + eps)
    else:
        is_off = ~np.array(state_array).astype(bool)
        
    padded = np.pad(is_off.astype(int), (1, 1), mode='constant', constant_values=0)
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return ends - starts

# ADDED HELPER: Calculates total time spent in off state
def get_total_off_time(state_array, agent_type=None, off_r=6.0, eps=0.01):
    if agent_type in ["jitter", "off_time"]:
        is_off = (state_array > off_r - eps) & (state_array < off_r + eps)
    else:
        is_off = ~np.array(state_array).astype(bool)
    return np.sum(is_off)

# ---------------------------------------------------------
# 3. Main Extraction Loop
# ---------------------------------------------------------
print("Extracting trend data across all capacitor sizes...")

for c in cap_sizes:
    V_thresh = thresh_map[c]
    capSizeStr = str(int(c)) if c.is_integer() else str(c)
    V_thresh_str = str(V_thresh)
    
    base_suffix = f"ANTWERP_ORIGINAL_ADR_{payloadOption}_BYTES_{capSizeStr}"
    baseline_suffix_opt = f"{base_suffix}_FARAD_{V_thresh_str}V_THRESH"
    baseline_suffix_1_9 = f"{base_suffix}_FARAD_1.9V_THRESH"
    baseline_suffix_3_45 = f"{base_suffix}_FARAD_3.45V_THRESH"
    other_suffix = f"{base_suffix}FARAD"
    
    train_jitter_base = f"JITTER_OPT_{training_neg_inaction_reward}_INACTION_REWARD_ORIGINAL_ADR_{trainingPayload}_BYTES_{trainingCapSizeStr}FARAD_TSF_{tsf_max}_GAMMA_{gamma}"
    infer_jitter_base = f"ANTWERP_JITTER_OPT_{training_neg_inaction_reward}_IN_REW_ORIGINAL_ADR_{payloadOption}_B_{capSizeStr}F_TSF_{tsf_max}"
    
    train_off_base = f"OFF_OPT_{training_neg_inaction_reward}_INACTION_REWARD_ORIGINAL_ADR_{trainingPayload}_BYTES_{trainingCapSizeStr}FARAD_TSF_{tsf_max}_GAMMA_{gamma}"
    infer_off_base = f"ANTWERP_OFF_OPT_{training_neg_inaction_reward}_IN_REW_ORIGINAL_ADR_{payloadOption}_B_{capSizeStr}F_TSF_{tsf_max}"

    folder = 'RLModelData/experiments_solar/inference_experiments'
    
    # --- A. Process Seeded RL Agents ---
    for seed in seeds:
        # Default placeholder in case load fails
        metrics_seed = {
            'jitter': {k: np.nan for k in metric_keys},
            'off_time': {k: np.nan for k in metric_keys}
        }
        try:
            ag_jit = load_agent_data(f'{folder}/rewardList{infer_jitter_base}____{train_jitter_base}_SEED_{seed}.csv')
            # Fallback to try without extra underscore if file not found
            if len(ag_jit) == 0:
                ag_jit = load_agent_data(f'{folder}/rewardList{infer_jitter_base}__{train_jitter_base}_SEED_{seed}.csv')
                
            ag_off = load_agent_data(f'{folder}/rewardList{infer_off_base}____{train_off_base}_SEED_{seed}.csv')
            if len(ag_off) == 0:
                ag_off = load_agent_data(f'{folder}/rewardList{infer_off_base}__{train_off_base}_SEED_{seed}.csv')

            if len(ag_jit) > 0:
                tbs = get_tbs(ag_jit, agent_type="jitter")
                metrics_seed['jitter']['mean_iti'] = np.mean(tbs) if len(tbs) > 0 else np.nan
                
                daily = get_daily_success_counts(ag_jit, agent_type="jitter")
                metrics_seed['jitter']['mean_daily_success'] = np.mean(daily) if len(daily) > 0 else 0
                
                surv = get_survival_time(ag_jit, agent_type="jitter")
                metrics_seed['jitter']['median_survival'] = np.median(surv) if len(surv) > 0 else 0
                
                daily_max = get_daily_max_iti(ag_jit, agent_type="jitter")
                metrics_seed['jitter']['median_max_daily_iti'] = np.median(daily_max) if len(daily_max) > 0 else np.nan
                
                b_dur = get_off_time_durations(ag_jit, agent_type="jitter")
                metrics_seed['jitter']['median_off_time_duration'] = np.median(b_dur) if len(b_dur) > 0 else np.nan
                
                # EXTRACT TOTAL OFF-TIME
                metrics_seed['jitter']['total_off_time'] = get_total_off_time(ag_jit, agent_type="jitter")

            if len(ag_off) > 0:
                tbs = get_tbs(ag_off, agent_type="off_time")
                metrics_seed['off_time']['mean_iti'] = np.mean(tbs) if len(tbs) > 0 else np.nan
                
                daily = get_daily_success_counts(ag_off, agent_type="off_time")
                metrics_seed['off_time']['mean_daily_success'] = np.mean(daily) if len(daily) > 0 else 0
                
                surv = get_survival_time(ag_off, agent_type="off_time")
                metrics_seed['off_time']['median_survival'] = np.median(surv) if len(surv) > 0 else 0
                
                daily_max = get_daily_max_iti(ag_off, agent_type="off_time")
                metrics_seed['off_time']['median_max_daily_iti'] = np.median(daily_max) if len(daily_max) > 0 else np.nan
                
                b_dur = get_off_time_durations(ag_off, agent_type="off_time")
                metrics_seed['off_time']['median_off_time_duration'] = np.median(b_dur) if len(b_dur) > 0 else np.nan

                # EXTRACT TOTAL OFF-TIME
                metrics_seed['off_time']['total_off_time'] = get_total_off_time(ag_off, agent_type="off_time")

        except Exception as e:
            print(f"Skipping seed {seed} for cap size {c}: {e}")

        for metric in metric_keys:
            trend_data_seeds[metric]['Agent (ITI)'][seed].append(metrics_seed['jitter'][metric])
            trend_data_seeds[metric]['Agent (Off-Time)'][seed].append(metrics_seed['off_time'][metric])

    # --- B. Process Baselines (Deterministic) ---
    try:
        b_succ = np.loadtxt(f'staticThreshModelData/successList{baseline_suffix_opt}.csv', delimiter='\t')
        b_state = pd.read_csv(f"staticThreshModelData/deviceStateList{baseline_suffix_opt}.csv", header=None).iloc[:, 0].astype(bool)
        
        b_succ_1_9 = np.loadtxt(f'staticThreshModelData/successList{baseline_suffix_1_9}.csv', delimiter='\t')
        b_state_1_9 = pd.read_csv(f"staticThreshModelData/deviceStateList{baseline_suffix_1_9}.csv", header=None).iloc[:, 0].astype(bool)

        b_succ_3_45 = np.loadtxt(f'staticThreshModelData/successList{baseline_suffix_3_45}.csv', delimiter='\t')
        b_state_3_45 = pd.read_csv(f"staticThreshModelData/deviceStateList{baseline_suffix_3_45}.csv", header=None).iloc[:, 0].astype(bool)

        p_succ = np.loadtxt(f'APModelData/successList{other_suffix}.csv', delimiter='\t')
        p_state = pd.read_csv(f"APModelData/deviceStateList{other_suffix}.csv", header=None).iloc[:, 0].astype(bool)
        
        o_succ = np.loadtxt(f'oracleModelData/successList{other_suffix}.csv', delimiter='\t')
        o_state = pd.read_csv(f"oracleModelData/deviceStateList{other_suffix}.csv", header=None).iloc[:, 0].astype(bool)
        
        a_succ = np.loadtxt(f'astarModelData/successList{other_suffix}.csv', delimiter='\t')
        a_state = pd.read_csv(f"astarModelData/deviceStateList{other_suffix}.csv", header=None).iloc[:, 0].astype(bool)

        succ_arrays = [b_succ, b_succ_1_9, b_succ_3_45, p_succ, o_succ, a_succ]
        state_arrays = [b_state, b_state_1_9, b_state_3_45, p_state, o_state, a_state]

        for i, label in enumerate(baseline_labels):
            tbs = get_tbs(succ_arrays[i], agent_type=None)
            trend_data_baselines['mean_iti'][label].append(np.mean(tbs) if len(tbs) > 0 else np.nan)
            
            daily = get_daily_success_counts(succ_arrays[i], agent_type=None)
            trend_data_baselines['mean_daily_success'][label].append(np.mean(daily) if len(daily) > 0 else 0)
            
            surv = get_survival_time(state_arrays[i], agent_type=None)
            trend_data_baselines['median_survival'][label].append(np.median(surv) if len(surv) > 0 else 0)

            daily_max = get_daily_max_iti(succ_arrays[i], agent_type=None)
            trend_data_baselines['median_max_daily_iti'][label].append(np.median(daily_max) if len(daily_max) > 0 else np.nan)
            
            b_dur = get_off_time_durations(state_arrays[i], agent_type=None)
            trend_data_baselines['median_off_time_duration'][label].append(np.median(b_dur) if len(b_dur) > 0 else np.nan)
            
            # EXTRACT TOTAL OFF-TIME
            total_off = get_total_off_time(state_arrays[i], agent_type=None)
            trend_data_baselines['total_off_time'][label].append(total_off if total_off is not None else np.nan)

    except Exception as e:
        print(f"Skipping baseline for cap size {c} due to missing data: {e}")
        for label in baseline_labels:
            for metric in metric_keys:
                trend_data_baselines[metric][label].append(np.nan)

print("Data extraction complete. Generating plots...")

# ---------------------------------------------------------
# 4. Plotting RL-agent means across seeds with +/- 1 standard deviation
# ---------------------------------------------------------
out_dir = "graphs_antwerp_original_adr_mean_std/comparison_all/trend_lines"
os.makedirs(out_dir, exist_ok=True)

LOG_FLOOR = 1e-1  # Minimum displayed value for logarithmic plots

def plot_trend(metric_key, title, ylabel, filename, use_log=False, ymin=None, ymax=None):
    plt.figure(figsize=(10, 6))

    # 1. Plot RL-agent mean across seeds, with a +/- 1 standard deviation band.
    for label in agent_labels:
        c_code = colors_dict[label]
        marker = markers_dict[label]
        ls = linestyles_dict[label]

        # Shape: (number of seeds, number of capacitor sizes)
        seed_matrix = np.array([
            trend_data_seeds[metric_key][label][seed] for seed in seeds
        ], dtype=float)

        means = np.full(len(cap_sizes), np.nan)
        stds = np.full(len(cap_sizes), np.nan)

        for cap_idx in range(len(cap_sizes)):
            valid_vals = seed_matrix[:, cap_idx]
            valid_vals = valid_vals[np.isfinite(valid_vals)]

            if len(valid_vals) > 0:
                means[cap_idx] = np.mean(valid_vals)
                # Use sample standard deviation; for a single seed, show no spread.
                stds[cap_idx] = np.std(valid_vals, ddof=1) if len(valid_vals) > 1 else 0.0

        valid_mask = np.isfinite(means) & np.isfinite(stds)
        x_valid = np.asarray(cap_sizes)[valid_mask]
        mean_valid = means[valid_mask]
        std_valid = stds[valid_mask]

        if len(x_valid) == 0:
            continue

        # For logarithmic plots, clip displayed values to the positive floor.
        if use_log:
            mean_valid = np.maximum(mean_valid, LOG_FLOOR)

        plt.plot(
            x_valid, mean_valid,
            label=label,
            color=c_code,
            marker=marker,
            linestyle=ls,
            linewidth=2.0,
            markersize=5,
            alpha=0.95
        )

        lower = mean_valid - std_valid
        upper = mean_valid + std_valid
        if use_log:
            lower = np.maximum(lower, LOG_FLOOR)
            upper = np.maximum(upper, LOG_FLOOR)

        plt.fill_between(
            x_valid, lower, upper,
            color=c_code,
            alpha=0.18,
            linewidth=0
        )

    # 2. Plot deterministic baseline models (unchanged).
    for label in baseline_labels:
        y_vals = np.array(trend_data_baselines[metric_key][label])
        valid_mask = ~np.isnan(y_vals)
        x_valid = np.array(cap_sizes)[valid_mask]
        y_valid = y_vals[valid_mask]

        if use_log:
            y_valid = np.maximum(y_valid, LOG_FLOOR)

        if len(x_valid) == 0:
            continue

        plt.plot(
            x_valid, y_valid,
            label=label,
            color=colors_dict[label],
            marker=markers_dict[label],
            linestyle=linestyles_dict[label],
            linewidth=2.5,
            markersize=7,
            alpha=0.9
        )

    if use_log:
        plt.yscale('log')
        ymin = LOG_FLOOR if ymin is None else max(ymin, LOG_FLOOR)

    if ymin is not None or ymax is not None:
        plt.ylim(bottom=ymin, top=ymax)

    plt.title(title, fontsize=14, pad=12)
    plt.xlabel('Capacitor Size (Farads)', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

    plt.xticks(cap_sizes, rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)

    plt.legend(loc='best', fontsize=11, framealpha=0.85)
    plt.tight_layout()

    filepath = os.path.join(out_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved trend plot: {filepath}")

# 1. Mean ITI 
plot_trend('mean_iti', 'Trend: Mean Inter-Task Interval vs. Capacitor Size (All Seeds)', 
           'Mean Time Steps Between Successes', 'trend_mean_iti_all_seeds.pdf', use_log=False, ymin=0)

# 2. Mean Daily Successes
plot_trend('mean_daily_success', 'Trend: Mean Daily Executions vs. Capacitor Size (All Seeds)', 
           'Mean Successful Transmissions per 24h', 'trend_daily_success_all_seeds.pdf', use_log=False)

# 3. Median Time Between Off-times
plot_trend('median_survival', 'Trend: Mean Time Between Off-States vs. Capacitor Size (Mean +/- SD across Seeds)', 
           'Mean Continuous ON-State Duration (Steps, Log)', 'trend_time_between_off_states_all_seeds.pdf', use_log=True)

# 4. Median Daily Max ITI 
plot_trend('median_max_daily_iti', 'Trend: Mean Daily Max ITI vs. Capacitor Size (Mean +/- SD across Seeds)', 
           'Mean Max Daily ITI (Steps, Log)', 'trend_max_daily_iti_all_seeds.pdf', use_log=True, ymin=100)

# 5. Median Off-time Duration
plot_trend('median_off_time_duration', 'Trend: Mean Off-State Duration vs. Capacitor Size (Mean +/- SD across Seeds)', 
           'Mean Continuous OFF-State Duration (Steps, Log)', 'trend_off_state_duration_all_seeds.pdf', use_log=True)

# 6. Total Off-time (NEW)
plot_trend('total_off_time', 'Trend: Total Off-Time vs. Capacitor Size (Mean +/- SD across Seeds)', 
           'Total Off-Time (Steps, Log)', 'trend_total_off_time_all_seeds.pdf', use_log=True)

print("All 6 trend plots (RL-agent mean +/- standard deviation across seeds) generated successfully as PDFs!")