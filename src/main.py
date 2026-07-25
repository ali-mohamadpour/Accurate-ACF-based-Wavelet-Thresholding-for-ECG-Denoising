# ============================================================
# Accurate ACF-based Wavelet Thresholding for ECG Denoising
# Paper:
# Yu et al. (2024)
# "Accurate wavelet thresholding method for ECG signals"
#
# Phase 1:
# - Read ECG/noise records
# - Validate dataset paths
# - Plot a small segment of MIT-BIH record 109
# ============================================================
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import wfdb
import pywt
from scipy import signal as scipy_signal


# ============================================================
# 1. Project paths and global configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "datasets"
MITDB_DIR = DATASET_DIR / "mitdb"
CUDB_DIR = DATASET_DIR / "cudb"
PTB_DIR = DATASET_DIR / "ptb"
NSTDB_DIR = DATASET_DIR / "nstdb"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
RESULT_DIR = OUTPUT_DIR / "results"
SIGNAL_DIR = OUTPUT_DIR / "signals"

for directory in [OUTPUT_DIR, FIGURE_DIR, RESULT_DIR, SIGNAL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ============================================================
# 2. Utility functions
# ============================================================

def print_section(title: str) -> None:
    """Print a visually clear section title in terminal."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def check_record_files(record_path: Path) -> bool:

    dat_file = record_path.with_suffix(".dat")
    hea_file = record_path.with_suffix(".hea")

    exists = dat_file.exists() and hea_file.exists()

    if not exists:
        print(f"[NOT FOUND] {record_path}")
        print(f"  Required: {dat_file.name} and {hea_file.name}")

    return exists


def load_wfdb_signal(record_path: Path, channel: int = 0):

    record = wfdb.rdrecord(str(record_path))

    if channel >= record.p_signal.shape[1]:
        raise ValueError(
            f"Requested channel {channel}, but record has only "
            f"{record.p_signal.shape[1]} channel(s)."
        )

    signal = record.p_signal[:, channel].astype(np.float64)
    fs = float(record.fs)

    return signal, fs, record


def print_record_info(name: str, signal: np.ndarray, fs: float, record) -> None:
    duration_seconds = len(signal) / fs

    print(f"Record name       : {name}")
    print(f"Sampling rate     : {fs:.2f} Hz")
    print(f"Number of samples : {len(signal)}")
    print(f"Duration          : {duration_seconds:.2f} s")
    print(f"Channels          : {record.n_sig}")
    print(f"Signal names      : {record.sig_name}")
    print(f"Signal min / max  : {np.min(signal):.5f} / {np.max(signal):.5f}")
    print(f"Signal mean / std : {np.mean(signal):.5f} / {np.std(signal):.5f}")


def plot_signal_segment(
    signal: np.ndarray,
    fs: float,
    title: str,
    output_filename: str,
    start_second: float = 0.0,
    duration_second: float = 10.0,
) -> None:

    start_index = int(start_second * fs)
    end_index = int((start_second + duration_second) * fs)

    end_index = min(end_index, len(signal))

    segment = signal[start_index:end_index]
    time_axis = np.arange(start_index, end_index) / fs

    plt.figure(figsize=(14, 4))
    plt.plot(time_axis, segment, color="navy", linewidth=0.9)
    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = FIGURE_DIR / output_filename
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"[SAVED FIGURE] {output_path}")


# ============================================================
# 3. Dataset validation
# ============================================================

def validate_datasets() -> None:

    print_section("DATASET PATH VALIDATION")

    records_to_check = {
        "MIT-BIH record 109": MITDB_DIR / "109",
        "MIT-BIH record 233": MITDB_DIR / "233",
        "CUDB record cu07": CUDB_DIR / "cu07",
        "CUDB record cu11": CUDB_DIR / "cu11",
        "PTB record s0016lre": PTB_DIR / "s0016lre",
        "PTB record s0026lre": PTB_DIR / "s0026lre",
        "NSTDB baseline wander standard name (bw)": NSTDB_DIR / "bw",
        "NSTDB electrode motion (em)": NSTDB_DIR / "em",
        "NSTDB muscle artifact (ma)": NSTDB_DIR / "ma",
    }

    for name, record_path in records_to_check.items():
        found = check_record_files(record_path)
        status = "OK" if found else "MISSING"
        print(f"{status:8s} | {name}")


# ============================================================
# 4. Initial loading experiment
# ============================================================

def run_initial_data_test() -> None:
    """
    Load MIT-BIH record 109 and plot its first 10 seconds.
    """
    print_section("INITIAL ECG LOADING TEST: MIT-BIH RECORD 109")

    record_path = MITDB_DIR / "109"

    if not check_record_files(record_path):
        print("Record 109 is unavailable. Check folder names and files.")
        return

    signal, fs, record = load_wfdb_signal(record_path, channel=0)

    print_record_info(
        name="MIT-BIH Arrhythmia Database - Record 109 - Channel 0",
        signal=signal,
        fs=fs,
        record=record,
    )

    plot_signal_segment(
        signal=signal,
        fs=fs,
        title="MIT-BIH Record 109 - Channel 0 - First 10 Seconds",
        output_filename="phase1_mitdb_109_first_10_seconds.png",
        start_second=0.0,
        duration_second=10.0,
    )

# ============================================================
# 5. Signal preparation, AWGN generation, and metrics
# ============================================================

def extract_signal_segment(
    signal: np.ndarray,
    fs: float,
    start_second: float,
    duration_second: float,
) -> np.ndarray:

    start_index = int(start_second * fs)
    end_index = int((start_second + duration_second) * fs)

    if start_index < 0:
        raise ValueError("start_second must be non-negative.")

    if end_index > len(signal):
        raise ValueError(
            f"Requested segment ends at sample {end_index}, "
            f"but signal length is only {len(signal)}."
        )

    return signal[start_index:end_index].copy()


def calculate_snr(clean_signal: np.ndarray, test_signal: np.ndarray) -> float:

    if len(clean_signal) != len(test_signal):
        raise ValueError("Signals must have the same length for SNR calculation.")

    error = test_signal - clean_signal

    signal_power = np.sum(clean_signal ** 2)
    error_power = np.sum(error ** 2)

    # Avoid division by zero in case the two signals are exactly equal
    error_power = max(error_power, np.finfo(np.float64).eps)

    snr_value = 10.0 * np.log10(signal_power / error_power)
    return float(snr_value)


def calculate_rmse(clean_signal: np.ndarray, test_signal: np.ndarray) -> float:

    if len(clean_signal) != len(test_signal):
        raise ValueError("Signals must have the same length for RMSE calculation.")

    return float(np.sqrt(np.mean((test_signal - clean_signal) ** 2)))



def add_awgn_at_target_snr(
    clean_signal: np.ndarray,
    target_snr_db: float,
    rng: np.random.Generator,
):

    clean_signal = clean_signal.astype(np.float64)

    # ECG average power
    signal_power = np.mean(clean_signal ** 2)

    # Required noise power for target input SNR
    target_noise_power = signal_power / (10.0 ** (target_snr_db / 10.0))

    # First create standard normal noise: mean=0 and variance≈1
    generated_noise = rng.normal(loc=0.0, scale=1.0, size=len(clean_signal))

    # Scale it so that its power precisely matches target_noise_power
    current_noise_power = np.mean(generated_noise ** 2)
    generated_noise = generated_noise * np.sqrt(
        target_noise_power / current_noise_power
    )

    noisy_signal = clean_signal + generated_noise

    actual_input_snr_db = calculate_snr(clean_signal, noisy_signal)

    return noisy_signal, generated_noise, actual_input_snr_db


def plot_clean_noisy_comparison(
    clean_signal: np.ndarray,
    noisy_signal: np.ndarray,
    fs: float,
    target_snr_db: float,
    output_filename: str,
    display_duration_second: float = 10.0,
) -> None:

    number_of_samples = min(
        int(display_duration_second * fs),
        len(clean_signal),
    )

    time_axis = np.arange(number_of_samples) / fs

    plt.figure(figsize=(15, 6))

    plt.subplot(2, 1, 1)
    plt.plot(
        time_axis,
        clean_signal[:number_of_samples],
        color="green",
        linewidth=0.9,
        label="Clean ECG",
    )
    plt.title("Clean ECG Signal")
    plt.ylabel("Amplitude (mV)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")

    plt.subplot(2, 1, 2)
    plt.plot(
        time_axis,
        noisy_signal[:number_of_samples],
        color="crimson",
        linewidth=0.7,
        label=f"Noisy ECG (Target input SNR = {target_snr_db} dB)",
    )
    plt.title("ECG Signal Corrupted by Additive White Gaussian Noise")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude (mV)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")

    plt.tight_layout()

    output_path = FIGURE_DIR / output_filename
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"[SAVED FIGURE] {output_path}")


def run_awgn_generation_test() -> None:

    print_section("PHASE 2: AWGN GENERATION AND METRIC VALIDATION")

    record_path = MITDB_DIR / "109"

    if not check_record_files(record_path):
        print("Record 109 is unavailable. AWGN test cannot run.")
        return

    full_signal, fs, _ = load_wfdb_signal(record_path, channel=0)

    clean_ecg = extract_signal_segment(
        signal=full_signal,
        fs=fs,
        start_second=0.0,
        duration_second=30.0,
    )

    target_snr_db = 5.0
    rng = np.random.default_rng(RANDOM_SEED)

    noisy_ecg, awgn_noise, actual_snr_db = add_awgn_at_target_snr(
        clean_signal=clean_ecg,
        target_snr_db=target_snr_db,
        rng=rng,
    )

    input_rmse = calculate_rmse(clean_ecg, noisy_ecg)

    print(f"ECG segment duration      : {len(clean_ecg) / fs:.2f} seconds")
    print(f"ECG segment samples       : {len(clean_ecg)}")
    print(f"Target input SNR          : {target_snr_db:.4f} dB")
    print(f"Actual input SNR          : {actual_snr_db:.4f} dB")
    print(f"Input RMSE                : {input_rmse:.6f}")
    print(f"AWGN mean                 : {np.mean(awgn_noise):.8f}")
    print(f"AWGN standard deviation   : {np.std(awgn_noise):.8f}")

    plot_clean_noisy_comparison(
        clean_signal=clean_ecg,
        noisy_signal=noisy_ecg,
        fs=fs,
        target_snr_db=target_snr_db,
        output_filename="phase2_clean_vs_awgn_5db_record109.png",
        display_duration_second=10.0,
    )

    np.save(SIGNAL_DIR / "record109_clean_30s.npy", clean_ecg)
    np.save(SIGNAL_DIR / "record109_awgn_5db_30s.npy", noisy_ecg)
    np.save(SIGNAL_DIR / "record109_awgn_noise_5db_30s.npy", awgn_noise)

    print(f"[SAVED SIGNAL] {SIGNAL_DIR / 'record109_clean_30s.npy'}")
    print(f"[SAVED SIGNAL] {SIGNAL_DIR / 'record109_awgn_5db_30s.npy'}")
# ============================================================
# 6. Wavelet decomposition and thresholding
# ============================================================

def estimate_noise_sigma_mad(detail_coefficients: np.ndarray) -> float:

    sigma = np.median(np.abs(detail_coefficients)) / 0.6745
    return float(sigma)


def calculate_universal_threshold(
    detail_coefficients: np.ndarray,
    signal_length: int,
) -> float:

    sigma = estimate_noise_sigma_mad(detail_coefficients)

    threshold = sigma * np.sqrt(2.0 * np.log(signal_length))
    return float(threshold)


def hard_threshold(coefficients: np.ndarray, threshold: float) -> np.ndarray:

    coefficients = np.asarray(coefficients, dtype=np.float64)

    return np.where(
        np.abs(coefficients) >= threshold,
        coefficients,
        0.0,
    )


def soft_threshold(coefficients: np.ndarray, threshold: float) -> np.ndarray:

    coefficients = np.asarray(coefficients, dtype=np.float64)

    return np.sign(coefficients) * np.maximum(
        np.abs(coefficients) - threshold,
        0.0,
    )


def wavelet_denoise_with_fixed_threshold(
    noisy_signal: np.ndarray,
    wavelet_name: str,
    decomposition_level: int,
    threshold: float,
    threshold_mode: str = "soft",
) -> np.ndarray:

    if threshold_mode not in {"soft", "hard"}:
        raise ValueError("threshold_mode must be either 'soft' or 'hard'.")

    wavelet = pywt.Wavelet(wavelet_name)

    max_allowed_level = pywt.dwt_max_level(
        data_len=len(noisy_signal),
        filter_len=wavelet.dec_len,
    )

    if decomposition_level > max_allowed_level:
        raise ValueError(
            f"Requested decomposition level is {decomposition_level}, "
            f"but the maximum allowed level is {max_allowed_level}."
        )

 
    coefficients = pywt.wavedec(
        data=noisy_signal,
        wavelet=wavelet_name,
        level=decomposition_level,
        mode="symmetric",
    )

    thresholded_coefficients = [coefficients[0]]

    for detail_coefficients in coefficients[1:]:
        if threshold_mode == "soft":
            processed_detail = soft_threshold(detail_coefficients, threshold)
        else:
            processed_detail = hard_threshold(detail_coefficients, threshold)

        thresholded_coefficients.append(processed_detail)

    denoised_signal = pywt.waverec(
        thresholded_coefficients,
        wavelet=wavelet_name,
        mode="symmetric",
    )


    denoised_signal = denoised_signal[:len(noisy_signal)]

    return denoised_signal


def plot_denoising_result(
    clean_signal: np.ndarray,
    noisy_signal: np.ndarray,
    denoised_signal: np.ndarray,
    fs: float,
    input_snr_db: float,
    output_snr_db: float,
    output_rmse: float,
    output_filename: str,
    display_duration_second: float = 10.0,
) -> None:

    number_of_samples = min(
        int(display_duration_second * fs),
        len(clean_signal),
    )

    time_axis = np.arange(number_of_samples) / fs

    plt.figure(figsize=(15, 8))

    plt.subplot(3, 1, 1)
    plt.plot(
        time_axis,
        clean_signal[:number_of_samples],
        color="green",
        linewidth=0.9,
    )
    plt.title("Clean ECG Signal")
    plt.ylabel("Amplitude (mV)")
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 2)
    plt.plot(
        time_axis,
        noisy_signal[:number_of_samples],
        color="crimson",
        linewidth=0.65,
    )
    plt.title(f"Noisy ECG Signal — Input SNR = {input_snr_db:.2f} dB")
    plt.ylabel("Amplitude (mV)")
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 3)
    plt.plot(
        time_axis,
        denoised_signal[:number_of_samples],
        color="navy",
        linewidth=0.85,
    )
    plt.title(
        "DWT Denoised ECG — "
        f"Output SNR = {output_snr_db:.2f} dB, "
        f"RMSE = {output_rmse:.5f}"
    )
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude (mV)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = FIGURE_DIR / output_filename
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"[SAVED FIGURE] {output_path}")


def run_wavelet_pipeline_test() -> None:

    print_section("PHASE 3: DWT AND SOFT-THRESHOLDING PIPELINE TEST")

    clean_path = SIGNAL_DIR / "record109_clean_30s.npy"
    noisy_path = SIGNAL_DIR / "record109_awgn_5db_30s.npy"


    if clean_path.exists() and noisy_path.exists():
        clean_ecg = np.load(clean_path)
        noisy_ecg = np.load(noisy_path)
        print("Loaded clean/noisy ECG signals saved in Phase 2.")
    else:
        print("Phase 2 files not found; regenerating the 5 dB AWGN test signal.")

        record_path = MITDB_DIR / "109"
        full_signal, fs, _ = load_wfdb_signal(record_path, channel=0)

        clean_ecg = extract_signal_segment(
            signal=full_signal,
            fs=fs,
            start_second=0.0,
            duration_second=30.0,
        )

        rng = np.random.default_rng(RANDOM_SEED)

        noisy_ecg, _, _ = add_awgn_at_target_snr(
            clean_signal=clean_ecg,
            target_snr_db=5.0,
            rng=rng,
        )

 
    _, fs, _ = load_wfdb_signal(MITDB_DIR / "109", channel=0)

    wavelet_name = "db3"
    decomposition_level = 4
    threshold_mode = "soft"

    coefficients = pywt.wavedec(
        data=noisy_ecg,
        wavelet=wavelet_name,
        level=decomposition_level,
        mode="symmetric",
    )

    finest_detail_coefficients = coefficients[-1]

    universal_threshold = calculate_universal_threshold(
        detail_coefficients=finest_detail_coefficients,
        signal_length=len(noisy_ecg),
    )

    denoised_ecg = wavelet_denoise_with_fixed_threshold(
        noisy_signal=noisy_ecg,
        wavelet_name=wavelet_name,
        decomposition_level=decomposition_level,
        threshold=universal_threshold,
        threshold_mode=threshold_mode,
    )

    input_snr = calculate_snr(clean_ecg, noisy_ecg)
    input_rmse = calculate_rmse(clean_ecg, noisy_ecg)

    output_snr = calculate_snr(clean_ecg, denoised_ecg)
    output_rmse = calculate_rmse(clean_ecg, denoised_ecg)

    print(f"Wavelet basis             : {wavelet_name}")
    print(f"Decomposition level       : {decomposition_level}")
    print(f"Threshold function        : {threshold_mode}")
    print(f"Estimated noise sigma     : "
          f"{estimate_noise_sigma_mad(finest_detail_coefficients):.6f}")
    print(f"Universal threshold       : {universal_threshold:.6f}")
    print(f"Input SNR                 : {input_snr:.4f} dB")
    print(f"Input RMSE                : {input_rmse:.6f}")
    print(f"Output SNR                : {output_snr:.4f} dB")
    print(f"Output RMSE               : {output_rmse:.6f}")
    print(f"SNR improvement           : {output_snr - input_snr:.4f} dB")

    plot_denoising_result(
        clean_signal=clean_ecg,
        noisy_signal=noisy_ecg,
        denoised_signal=denoised_ecg,
        fs=fs,
        input_snr_db=input_snr,
        output_snr_db=output_snr,
        output_rmse=output_rmse,
        output_filename="phase3_dwt_universal_soft_record109_awgn5db.png",
        display_duration_second=10.0,
    )

    np.save(
        SIGNAL_DIR / "record109_denoised_universal_soft_5db_30s.npy",
        denoised_ecg,
    )

    print(
        "[SAVED SIGNAL] "
        f"{SIGNAL_DIR / 'record109_denoised_universal_soft_5db_30s.npy'}"
    )
# ============================================================
# 7. Classical thresholding methods:
#    Universal, Minimax, Heursure, BayesShrink
# ============================================================

def calculate_minimax_threshold(
    detail_coefficients: np.ndarray,
    signal_length: int,
) -> float:

    sigma = estimate_noise_sigma_mad(detail_coefficients)

    if signal_length <= 32:
        return 0.0

    threshold = sigma * (
        0.3936 + 0.1829 * np.log2(signal_length)
    )

    return float(threshold)


def calculate_sure_threshold_normalized(
    normalized_coefficients: np.ndarray,
) -> float:

    coefficients = np.asarray(
        normalized_coefficients,
        dtype=np.float64,
    ).ravel()

    number_of_coefficients = len(coefficients)

    if number_of_coefficients == 0:
        return 0.0

    squared_sorted = np.sort(np.abs(coefficients) ** 2)

    index = np.arange(1, number_of_coefficients + 1)

    risks = (
        number_of_coefficients
        - 2.0 * index
        + np.cumsum(squared_sorted)
        + (number_of_coefficients - index) * squared_sorted
    ) / number_of_coefficients

    minimum_risk_index = int(np.argmin(risks))

    sure_threshold = np.sqrt(squared_sorted[minimum_risk_index])

    return float(sure_threshold)


def calculate_heursure_threshold(
    detail_coefficients: np.ndarray,
    signal_length: int,
) -> float:

    sigma = estimate_noise_sigma_mad(detail_coefficients)

    if sigma <= np.finfo(np.float64).eps:
        return 0.0

    normalized_coefficients = detail_coefficients / sigma
    number_of_coefficients = len(normalized_coefficients)

    universal_normalized = np.sqrt(
        2.0 * np.log(max(number_of_coefficients, 2))
    )

    sure_normalized = calculate_sure_threshold_normalized(
        normalized_coefficients
    )

    energy_measure = (
        np.sum(normalized_coefficients ** 2)
        - number_of_coefficients
    ) / number_of_coefficients

    criterion = (
        np.log2(max(number_of_coefficients, 2)) ** 1.5
    ) / np.sqrt(number_of_coefficients)

    if energy_measure < criterion:
        selected_normalized_threshold = universal_normalized
    else:
        selected_normalized_threshold = min(
            sure_normalized,
            universal_normalized,
        )

    return float(sigma * selected_normalized_threshold)


def calculate_bayes_threshold(
    detail_coefficients: np.ndarray,
    noise_sigma: float,
) -> float:

    noise_variance = noise_sigma ** 2
    coefficient_variance = np.var(detail_coefficients)

    signal_variance = coefficient_variance - noise_variance

    if signal_variance <= np.finfo(np.float64).eps:
        return float(np.max(np.abs(detail_coefficients)) + 1.0)

    signal_sigma = np.sqrt(signal_variance)

    bayes_threshold = noise_variance / signal_sigma

    return float(bayes_threshold)


def wavelet_denoise_bayes(
    noisy_signal: np.ndarray,
    wavelet_name: str,
    decomposition_level: int,
    threshold_mode: str = "soft",
):

    if threshold_mode not in {"soft", "hard"}:
        raise ValueError("threshold_mode must be 'soft' or 'hard'.")

    coefficients = pywt.wavedec(
        data=noisy_signal,
        wavelet=wavelet_name,
        level=decomposition_level,
        mode="symmetric",
    )

    finest_detail_coefficients = coefficients[-1]

    noise_sigma = estimate_noise_sigma_mad(finest_detail_coefficients)

    processed_coefficients = [coefficients[0]]
    bayes_thresholds = []

    for detail_coefficients in coefficients[1:]:
        threshold = calculate_bayes_threshold(
            detail_coefficients=detail_coefficients,
            noise_sigma=noise_sigma,
        )

        bayes_thresholds.append(threshold)

        if threshold_mode == "soft":
            processed_detail = soft_threshold(
                detail_coefficients,
                threshold,
            )
        else:
            processed_detail = hard_threshold(
                detail_coefficients,
                threshold,
            )

        processed_coefficients.append(processed_detail)

    denoised_signal = pywt.waverec(
        processed_coefficients,
        wavelet=wavelet_name,
        mode="symmetric",
    )

    denoised_signal = denoised_signal[:len(noisy_signal)]

    return denoised_signal, bayes_thresholds


def plot_classical_methods_comparison(
    clean_signal: np.ndarray,
    noisy_signal: np.ndarray,
    denoised_signals: dict,
    fs: float,
    output_filename: str,
    display_duration_second: float = 10.0,
) -> None:

    number_of_samples = min(
        int(display_duration_second * fs),
        len(clean_signal),
    )

    time_axis = np.arange(number_of_samples) / fs

    method_colors = {
        "Universal": "tab:blue",
        "Minimax": "tab:orange",
        "Heursure": "tab:purple",
        "Bayes": "tab:red",
    }

    plt.figure(figsize=(16, 13))

    plt.subplot(6, 1, 1)
    plt.plot(
        time_axis,
        clean_signal[:number_of_samples],
        color="green",
        linewidth=0.9,
    )
    plt.title("Clean ECG Signal")
    plt.ylabel("mV")
    plt.grid(True, alpha=0.3)

    plt.subplot(6, 1, 2)
    plt.plot(
        time_axis,
        noisy_signal[:number_of_samples],
        color="black",
        linewidth=0.65,
    )
    plt.title("Noisy ECG Signal")
    plt.ylabel("mV")
    plt.grid(True, alpha=0.3)

    for plot_index, (method_name, result) in enumerate(
        denoised_signals.items(),
        start=3,
    ):
        denoised_ecg = result["signal"]
        snr = result["snr"]
        rmse = result["rmse"]

        plt.subplot(6, 1, plot_index)
        plt.plot(
            time_axis,
            denoised_ecg[:number_of_samples],
            color=method_colors.get(method_name, "navy"),
            linewidth=0.8,
        )
        plt.title(
            f"{method_name} Thresholding — "
            f"Output SNR = {snr:.3f} dB, RMSE = {rmse:.6f}"
        )
        plt.ylabel("mV")
        plt.grid(True, alpha=0.3)

    plt.xlabel("Time (seconds)")
    plt.tight_layout()

    output_path = FIGURE_DIR / output_filename
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"[SAVED FIGURE] {output_path}")


def run_classical_methods_comparison() -> None:

    print_section(
        "PHASE 4: CLASSICAL THRESHOLDING METHODS COMPARISON"
    )

    clean_path = SIGNAL_DIR / "record109_clean_30s.npy"
    noisy_path = SIGNAL_DIR / "record109_awgn_5db_30s.npy"

    if not clean_path.exists() or not noisy_path.exists():
        raise FileNotFoundError(
            "Phase 2 signal files were not found. "
            "Run Phase 2 before Phase 4."
        )

    clean_ecg = np.load(clean_path)
    noisy_ecg = np.load(noisy_path)

    _, fs, _ = load_wfdb_signal(MITDB_DIR / "109", channel=0)

    wavelet_name = "db3"
    decomposition_level = 4
    threshold_mode = "soft"

    coefficients = pywt.wavedec(
        data=noisy_ecg,
        wavelet=wavelet_name,
        level=decomposition_level,
        mode="symmetric",
    )

    finest_detail_coefficients = coefficients[-1]

    universal_threshold = calculate_universal_threshold(
        detail_coefficients=finest_detail_coefficients,
        signal_length=len(noisy_ecg),
    )

    minimax_threshold = calculate_minimax_threshold(
        detail_coefficients=finest_detail_coefficients,
        signal_length=len(noisy_ecg),
    )

    heursure_threshold = calculate_heursure_threshold(
        detail_coefficients=finest_detail_coefficients,
        signal_length=len(noisy_ecg),
    )

    universal_ecg = wavelet_denoise_with_fixed_threshold(
        noisy_signal=noisy_ecg,
        wavelet_name=wavelet_name,
        decomposition_level=decomposition_level,
        threshold=universal_threshold,
        threshold_mode=threshold_mode,
    )

    minimax_ecg = wavelet_denoise_with_fixed_threshold(
        noisy_signal=noisy_ecg,
        wavelet_name=wavelet_name,
        decomposition_level=decomposition_level,
        threshold=minimax_threshold,
        threshold_mode=threshold_mode,
    )

    heursure_ecg = wavelet_denoise_with_fixed_threshold(
        noisy_signal=noisy_ecg,
        wavelet_name=wavelet_name,
        decomposition_level=decomposition_level,
        threshold=heursure_threshold,
        threshold_mode=threshold_mode,
    )

    bayes_ecg, bayes_thresholds = wavelet_denoise_bayes(
        noisy_signal=noisy_ecg,
        wavelet_name=wavelet_name,
        decomposition_level=decomposition_level,
        threshold_mode=threshold_mode,
    )

    results = {
        "Universal": {
            "signal": universal_ecg,
            "threshold": universal_threshold,
        },
        "Minimax": {
            "signal": minimax_ecg,
            "threshold": minimax_threshold,
        },
        "Heursure": {
            "signal": heursure_ecg,
            "threshold": heursure_threshold,
        },
        "Bayes": {
            "signal": bayes_ecg,
            "threshold": bayes_thresholds,
        },
    }

    input_snr = calculate_snr(clean_ecg, noisy_ecg)
    input_rmse = calculate_rmse(clean_ecg, noisy_ecg)

    print(f"Wavelet basis             : {wavelet_name}")
    print(f"Decomposition level       : {decomposition_level}")
    print(f"Threshold function        : {threshold_mode}")
    print(f"Input SNR                 : {input_snr:.4f} dB")
    print(f"Input RMSE                : {input_rmse:.6f}")

    print("\n" + "-" * 78)
    print(
        f"{'Method':<12}"
        f"{'Threshold':<30}"
        f"{'Output SNR (dB)':>18}"
        f"{'RMSE':>14}"
    )
    print("-" * 78)

    for method_name, result in results.items():
        denoised_ecg = result["signal"]

        output_snr = calculate_snr(clean_ecg, denoised_ecg)
        output_rmse = calculate_rmse(clean_ecg, denoised_ecg)

        result["snr"] = output_snr
        result["rmse"] = output_rmse

        threshold_value = result["threshold"]

        if isinstance(threshold_value, list):
            threshold_text = (
                "["
                + ", ".join(f"{item:.4f}" for item in threshold_value)
                + "]"
            )
        else:
            threshold_text = f"{threshold_value:.6f}"

        print(
            f"{method_name:<12}"
            f"{threshold_text:<30}"
            f"{output_snr:>18.4f}"
            f"{output_rmse:>14.6f}"
        )

        np.save(
            SIGNAL_DIR / f"record109_{method_name.lower()}_soft_5db_30s.npy",
            denoised_ecg,
        )

    print("-" * 78)

    plot_classical_methods_comparison(
        clean_signal=clean_ecg,
        noisy_signal=noisy_ecg,
        denoised_signals=results,
        fs=fs,
        output_filename=(
            "phase4_classical_methods_record109_awgn5db.png"
        ),
        display_duration_second=10.0,
    )
def remove_baseline_wander(
    signal: np.ndarray,
    fs: float,
    cutoff_frequency: float = 0.5,
) -> np.ndarray:

    from scipy import signal as scipy_signal

    nyquist_frequency = fs / 2.0
    normalized_cutoff = cutoff_frequency / nyquist_frequency

    b, a = scipy_signal.butter(
        N=4,
        Wn=normalized_cutoff,
        btype="high",
    )

    filtered_signal = scipy_signal.filtfilt(b, a, signal)

    return filtered_signal

# ============================================================
# 8. ACF-based Adaptive Thresholding (Proposed Method)
# ============================================================

def calculate_normalized_acf(
    signal: np.ndarray,
    remove_dc: bool = True,
) -> np.ndarray:

    signal = np.asarray(signal, dtype=np.float64).ravel()
    signal_length = len(signal)

    if signal_length == 0:
        return np.array([])

    if remove_dc:
        signal_zero_mean = signal - np.mean(signal)
    else:
        signal_zero_mean = signal

    signal_power = np.sum(signal_zero_mean ** 2)

    if signal_power <= np.finfo(np.float64).eps:
        return np.zeros(signal_length)

    fft_length = 2 ** int(np.ceil(np.log2(2 * signal_length - 1)))

    signal_fft = np.fft.fft(signal_zero_mean, n=fft_length)
    acf_unscaled = np.fft.ifft(
        signal_fft * np.conj(signal_fft)
    ).real

    acf_unscaled = acf_unscaled[:signal_length]

    normalized_acf = acf_unscaled / signal_power

    return normalized_acf


def calculate_nzopp(
    normalized_acf: np.ndarray,
    min_lag: int = 1,
    max_lag: int = None,
    peak_threshold: float = 0.1,
) -> float:

    acf = np.asarray(normalized_acf, dtype=np.float64).ravel()
    signal_length = len(acf)

    if max_lag is None:
        max_lag = signal_length // 2

    if max_lag >= signal_length:
        max_lag = signal_length - 1

    if min_lag >= max_lag:
        return 0.0

    acf_segment = acf[min_lag:max_lag + 1]

    if len(acf_segment) == 0:
        return 0.0

    points_above_threshold = np.sum(acf_segment > peak_threshold)
    total_points = len(acf_segment)

    nzopp = points_above_threshold / total_points

    return float(nzopp)


def plot_acf_comparison_debug(
    clean_signal: np.ndarray,
    noisy_signal: np.ndarray,
    denoised_signals: dict,
    fs: float,
    output_filename: str,
    display_duration_second: float = 10.0,
) -> None:
    
    signals_to_plot = {
        "Clean ECG": clean_signal,
        "Noisy ECG": noisy_signal,
    }

    for method_name, result in denoised_signals.items():
        signals_to_plot[method_name] = result["signal"]

    plt.figure(figsize=(16, 12))

    for plot_index, (signal_name, signal_data) in enumerate(
        signals_to_plot.items(),
        start=1,
    ):
        acf = calculate_normalized_acf(signal_data)
        lags_to_show = min(500, len(acf) // 2)

        plt.subplot(len(signals_to_plot), 1, plot_index)
        plt.plot(
            np.arange(lags_to_show),
            acf[:lags_to_show],
            linewidth=0.8,
        )
        plt.axhline(y=0.5, color="red", linestyle="--", alpha=0.5,
                    label="Threshold = 0.5")
        plt.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        plt.title(f"Normalized ACF - {signal_name}")
        plt.xlabel("Lag")
        plt.ylabel("ACF")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper right")

    plt.tight_layout()

    output_path = FIGURE_DIR / output_filename
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"[SAVED FIGURE] {output_path}")


def evaluate_threshold_quality(
    noisy_signal: np.ndarray,
    wavelet_name: str,
    decomposition_level: int,
    threshold: float,
    threshold_mode: str = "soft",
    fs: float = 360.0,
    preprocess: bool = True,
) -> float:

    denoised_signal = wavelet_denoise_with_fixed_threshold(
        noisy_signal=noisy_signal,
        wavelet_name=wavelet_name,
        decomposition_level=decomposition_level,
        threshold=threshold,
        threshold_mode=threshold_mode,
    )

    if preprocess:
        denoised_signal = remove_baseline_wander(
            denoised_signal,
            fs=fs,
        )

    normalized_acf = calculate_normalized_acf(
        denoised_signal,
        remove_dc=True,
    )
    nzopp = calculate_nzopp(normalized_acf)

    return float(nzopp)


def fast_threshold_query(
    noisy_signal: np.ndarray,
    wavelet_name: str,
    decomposition_level: int,
    threshold_mode: str = "soft",
    initial_thresholds: list = None,
    convergence_tolerance: float = 1e-6,
    max_iterations: int = 20,
    verbose: bool = True,
    fs: float = 360.0,
    preprocess: bool = True,
) -> tuple:
    """
    Fast Threshold Querying Algorithm (Fig. 3 in the paper).
    """
    if initial_thresholds is None:
        initial_thresholds = [0.5, 2.0, 8.0]

    if len(initial_thresholds) != 3:
        raise ValueError("initial_thresholds must have exactly 3 values.")

    thresholds = sorted(initial_thresholds)

    threshold_history = []
    nzopp_history = []

    iteration_count = 0
    converged = False

    while iteration_count < max_iterations and not converged:
        iteration_count += 1

        nzopp_values = []
        for threshold in thresholds:
            nzopp = evaluate_threshold_quality(
                noisy_signal=noisy_signal,
                wavelet_name=wavelet_name,
                decomposition_level=decomposition_level,
                threshold=threshold,
                threshold_mode=threshold_mode,
                fs=fs,
                preprocess=preprocess,
            )
            nzopp_values.append(nzopp)

            threshold_history.append(threshold)
            nzopp_history.append(nzopp)

        max_nzopp_index = int(np.argmax(nzopp_values))
        max_nzopp = nzopp_values[max_nzopp_index]
        max_threshold = thresholds[max_nzopp_index]

        if verbose:
            print(
                f"  Iteration {iteration_count}: "
                f"thresholds = {[f'{t:.6f}' for t in thresholds]}, "
                f"NZOPP = {[f'{n:.4f}' for n in nzopp_values]}"
            )

        # Check convergence
        if len(thresholds) >= 2:
            threshold_diffs = np.abs(np.diff(thresholds))
            if np.all(threshold_diffs < convergence_tolerance):
                if verbose:
                    print(
                        f"  Converged after {iteration_count} iterations."
                    )
                converged = True
                break

        new_thresholds = []

        if max_nzopp_index > 0:
            mid_point = (
                thresholds[max_nzopp_index - 1] + max_threshold
            ) / 2.0
            new_thresholds.append(thresholds[max_nzopp_index - 1])
            new_thresholds.append(mid_point)

        new_thresholds.append(max_threshold)

        if max_nzopp_index < len(thresholds) - 1:
            mid_point = (
                max_threshold + thresholds[max_nzopp_index + 1]
            ) / 2.0
            new_thresholds.append(mid_point)
            new_thresholds.append(thresholds[max_nzopp_index + 1])

        thresholds = sorted(set(new_thresholds))

        if len(thresholds) > 7:
            best_index = thresholds.index(max_threshold)
            start_index = max(0, best_index - 3)
            end_index = min(len(thresholds), best_index + 4)
            thresholds = thresholds[start_index:end_index]

    if iteration_count == 0:
        nzopp_values = []
        for threshold in thresholds:
            nzopp = evaluate_threshold_quality(
                noisy_signal=noisy_signal,
                wavelet_name=wavelet_name,
                decomposition_level=decomposition_level,
                threshold=threshold,
                threshold_mode=threshold_mode,
                fs=fs,
                preprocess=preprocess,
            )
            nzopp_values.append(nzopp)

        max_nzopp_index = int(np.argmax(nzopp_values))
        max_nzopp = nzopp_values[max_nzopp_index]
        max_threshold = thresholds[max_nzopp_index]

    optimal_threshold = max_threshold
    optimal_nzopp = max_nzopp

    return (
        optimal_threshold,
        optimal_nzopp,
        iteration_count,
        threshold_history,
        nzopp_history,
    )

def wavelet_denoise_acf(
    noisy_signal: np.ndarray,
    wavelet_name: str,
    decomposition_level: int,
    threshold_mode: str = "soft",
    verbose: bool = True,
    fs: float = 360.0,
    preprocess: bool = True,
) -> tuple:

    result = fast_threshold_query(
        noisy_signal=noisy_signal,
        wavelet_name=wavelet_name,
        decomposition_level=decomposition_level,
        threshold_mode=threshold_mode,
        verbose=verbose,
        fs=fs,
        preprocess=preprocess,
    )

    if result is None:
        raise RuntimeError(
            "fast_threshold_query returned None."
        )

    optimal_threshold, optimal_nzopp, iteration_count, _, _ = result

    denoised_signal = wavelet_denoise_with_fixed_threshold(
        noisy_signal=noisy_signal,
        wavelet_name=wavelet_name,
        decomposition_level=decomposition_level,
        threshold=optimal_threshold,
        threshold_mode=threshold_mode,
    )

    return denoised_signal, optimal_threshold, optimal_nzopp, iteration_count


def plot_acf_comparison(
    clean_signal: np.ndarray,
    noisy_signal: np.ndarray,
    denoised_signals: dict,
    fs: float,
    output_filename: str,
    display_duration_second: float = 10.0,
) -> None:

    number_of_samples = min(
        int(display_duration_second * fs),
        len(clean_signal),
    )

    time_axis = np.arange(number_of_samples) / fs

    method_colors = {
        "Universal": "tab:blue",
        "Minimax": "tab:orange",
        "Heursure": "tab:purple",
        "Bayes": "tab:red",
        "ACF (Proposed)": "tab:green",
    }

    plt.figure(figsize=(16, 15))

    plt.subplot(7, 1, 1)
    plt.plot(
        time_axis,
        clean_signal[:number_of_samples],
        color="green",
        linewidth=0.9,
    )
    plt.title("Clean ECG Signal")
    plt.ylabel("mV")
    plt.grid(True, alpha=0.3)

    plt.subplot(7, 1, 2)
    plt.plot(
        time_axis,
        noisy_signal[:number_of_samples],
        color="black",
        linewidth=0.65,
    )
    plt.title("Noisy ECG Signal")
    plt.ylabel("mV")
    plt.grid(True, alpha=0.3)

    for plot_index, (method_name, result) in enumerate(
        denoised_signals.items(),
        start=3,
    ):
        denoised_ecg = result["signal"]
        snr = result["snr"]
        rmse = result["rmse"]

        plt.subplot(7, 1, plot_index)
        plt.plot(
            time_axis,
            denoised_ecg[:number_of_samples],
            color=method_colors.get(method_name, "navy"),
            linewidth=0.8,
        )
        plt.title(
            f"{method_name} — "
            f"Output SNR = {snr:.3f} dB, RMSE = {rmse:.6f}"
        )
        plt.ylabel("mV")
        plt.grid(True, alpha=0.3)

    plt.xlabel("Time (seconds)")
    plt.tight_layout()

    output_path = FIGURE_DIR / output_filename
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"[SAVED FIGURE] {output_path}")


def run_acf_method_comparison() -> None:
 
    print_section(
        "PHASE 5: ACF-BASED ADAPTIVE THRESHOLDING COMPARISON"
    )

    clean_path = SIGNAL_DIR / "record109_clean_30s.npy"
    noisy_path = SIGNAL_DIR / "record109_awgn_5db_30s.npy"

    if not clean_path.exists() or not noisy_path.exists():
        raise FileNotFoundError(
            "Phase 2 signal files were not found."
        )

    clean_ecg = np.load(clean_path)
    noisy_ecg = np.load(noisy_path)

    _, fs, _ = load_wfdb_signal(MITDB_DIR / "109", channel=0)

    wavelet_name = "db3"
    decomposition_level = 4
    threshold_mode = "soft"

    classical_results = {}

    for method_name in ["Universal", "Minimax", "Heursure", "Bayes"]:
        signal_path = (
            SIGNAL_DIR
            / f"record109_{method_name.lower()}_soft_5db_30s.npy"
        )

        if signal_path.exists():
            denoised_ecg = np.load(signal_path)
        else:
            print(
                f"Warning: {signal_path} not found. "
                f"Recalculating {method_name}..."
            )
            continue

        output_snr = calculate_snr(clean_ecg, denoised_ecg)
        output_rmse = calculate_rmse(clean_ecg, denoised_ecg)

        classical_results[method_name] = {
            "signal": denoised_ecg,
            "snr": output_snr,
            "rmse": output_rmse,
        }

    print("\nRunning ACF-based adaptive thresholding...")
    print("This may take a moment due to iterative threshold search.\n")

    acf_denoised, acf_threshold, acf_nzopp, acf_iterations = (
        wavelet_denoise_acf(
            noisy_signal=noisy_ecg,
            wavelet_name=wavelet_name,
            decomposition_level=decomposition_level,
            threshold_mode=threshold_mode,
            verbose=True,
        )
    )

    acf_output_snr = calculate_snr(clean_ecg, acf_denoised)
    acf_output_rmse = calculate_rmse(clean_ecg, acf_denoised)

    print(f"\nACF Method Results:")
    print(f"  Optimal threshold : {acf_threshold:.6f}")
    print(f"  Optimal NZOPP     : {acf_nzopp:.6f}")
    print(f"  Iterations        : {acf_iterations}")
    print(f"  Output SNR        : {acf_output_snr:.4f} dB")
    print(f"  Output RMSE       : {acf_output_rmse:.6f}")

    all_results = dict(classical_results)
    all_results["ACF (Proposed)"] = {
        "signal": acf_denoised,
        "snr": acf_output_snr,
        "rmse": acf_output_rmse,
    }

    print("\n" + "=" * 78)
    print(
        f"{'Method':<20}"
        f"{'Threshold':<20}"
        f"{'Output SNR (dB)':>18}"
        f"{'RMSE':>14}"
    )
    print("=" * 78)

    for method_name, result in all_results.items():
        output_snr = result["snr"]
        output_rmse = result["rmse"]

        if method_name == "ACF (Proposed)":
            threshold_text = f"{acf_threshold:.6f}"
        elif method_name == "Bayes":
            threshold_text = "per-subband"
        else:
            threshold_text = "fixed"

        print(
            f"{method_name:<20}"
            f"{threshold_text:<20}"
            f"{output_snr:>18.4f}"
            f"{output_rmse:>14.6f}"
        )

    print("=" * 78)

    np.save(
        SIGNAL_DIR / "record109_acf_soft_5db_30s.npy",
        acf_denoised,
    )

    table_path = RESULT_DIR / "phase5_comparison_table.txt"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("Comparison of Thresholding Methods\n")
        f.write("=" * 78 + "\n")
        f.write(
            f"{'Method':<20}"
            f"{'Threshold':<20}"
            f"{'Output SNR (dB)':>18}"
            f"{'RMSE':>14}\n"
        )
        f.write("=" * 78 + "\n")

        for method_name, result in all_results.items():
            output_snr = result["snr"]
            output_rmse = result["rmse"]

            if method_name == "ACF (Proposed)":
                threshold_text = f"{acf_threshold:.6f}"
            elif method_name == "Bayes":
                threshold_text = "per-subband"
            else:
                threshold_text = "fixed"

            f.write(
                f"{method_name:<20}"
                f"{threshold_text:<20}"
                f"{output_snr:>18.4f}"
                f"{output_rmse:>14.6f}\n"
            )

        f.write("=" * 78 + "\n")
        f.write(f"\nACF Method Details:\n")
        f.write(f"  Optimal threshold : {acf_threshold:.6f}\n")
        f.write(f"  Optimal NZOPP     : {acf_nzopp:.6f}\n")
        f.write(f"  Iterations        : {acf_iterations}\n")

    print(f"[SAVED TABLE] {table_path}")

    # Plot comparison
    plot_acf_comparison(
        clean_signal=clean_ecg,
        noisy_signal=noisy_ecg,
        denoised_signals=all_results,
        fs=fs,
        output_filename="phase5_acf_comparison_record109_awgn5db.png",
        display_duration_second=10.0,
    )
        # Plot ACF comparison for debugging
    plot_acf_comparison_debug(
        clean_signal=clean_ecg,
        noisy_signal=noisy_ecg,
        denoised_signals=all_results,
        fs=fs,
        output_filename="phase5_acf_debug_record109_awgn5db.png",
        display_duration_second=10.0,
    )
# ============================================================
# 9. Real noise experiments (Baseline Wander, Electrode Motion, Muscle Artifact)
# ============================================================

def load_noise_record(
    noise_path: Path,
    target_length: int,
    channel: int = 0,
) -> np.ndarray:

    record = wfdb.rdrecord(str(noise_path))
    noise_signal = record.p_signal[:, channel].astype(np.float64)

    # Truncate or pad to target length
    if len(noise_signal) >= target_length:
        noise_segment = noise_signal[:target_length]
    else:
        # Pad with zeros if noise is shorter than target
        noise_segment = np.pad(
            noise_signal,
            (0, target_length - len(noise_signal)),
            mode="wrap",
        )

    return noise_segment


def add_real_noise_at_target_snr(
    clean_signal: np.ndarray,
    noise_signal: np.ndarray,
    target_snr_db: float,
) -> tuple:

    clean_signal = clean_signal.astype(np.float64)
    noise_signal = noise_signal.astype(np.float64)

    min_length = min(len(clean_signal), len(noise_signal))
    clean_signal = clean_signal[:min_length]
    noise_signal = noise_signal[:min_length]

    noise_signal = noise_signal - np.mean(noise_signal)

    signal_power = np.mean(clean_signal ** 2)
    noise_power = np.mean(noise_signal ** 2)

    if noise_power <= np.finfo(np.float64).eps:
        return clean_signal.copy(), np.zeros_like(clean_signal), float("inf")

    target_noise_power = signal_power / (10.0 ** (target_snr_db / 10.0))

    scaling_factor = np.sqrt(target_noise_power / noise_power)
    scaled_noise = noise_signal * scaling_factor

    noisy_signal = clean_signal + scaled_noise

    actual_input_snr_db = calculate_snr(clean_signal, noisy_signal)

    return noisy_signal, scaled_noise, actual_input_snr_db


def run_real_noise_experiment() -> None:
 
    print_section(
        "PHASE 6: REAL NOISE EXPERIMENTS"
    )

    clean_path = SIGNAL_DIR / "record109_clean_30s.npy"

    if not clean_path.exists():
        record_path = MITDB_DIR / "109"
        full_signal, fs, _ = load_wfdb_signal(record_path, channel=0)
        clean_ecg = extract_signal_segment(
            signal=full_signal,
            fs=fs,
            start_second=0.0,
            duration_second=30.0,
        )
        np.save(clean_path, clean_ecg)
    else:
        clean_ecg = np.load(clean_path)

    _, fs, _ = load_wfdb_signal(MITDB_DIR / "109", channel=0)

    wavelet_name = "db3"
    decomposition_level = 4
    threshold_mode = "soft"

    noise_types = {
        "Baseline Wander": NSTDB_DIR / "bw",
        "Electrode Motion": NSTDB_DIR / "em",
        "Muscle Artifact": NSTDB_DIR / "ma",
    }

    all_results = {}

    for noise_name, noise_path in noise_types.items():
        print(f"\n{'=' * 60}")
        print(f"NOISE TYPE: {noise_name}")
        print(f"{'=' * 60}")

        noise_signal = load_noise_record(
            noise_path=noise_path,
            target_length=len(clean_ecg),
            channel=0,
        )

        target_input_snr = 0.0
        noisy_ecg, scaled_noise, actual_input_snr = (
            add_real_noise_at_target_snr(
                clean_signal=clean_ecg,
                noise_signal=noise_signal,
                target_snr_db=target_input_snr,
            )
        )

        print(f"Target input SNR : {target_input_snr:.1f} dB")
        print(f"Actual input SNR : {actual_input_snr:.2f} dB")

        noise_safe_name = noise_name.lower().replace(" ", "_")
        np.save(
            SIGNAL_DIR / f"record109_{noise_safe_name}_noisy.npy",
            noisy_ecg,
        )

        coefficients = pywt.wavedec(
            data=noisy_ecg,
            wavelet=wavelet_name,
            level=decomposition_level,
            mode="symmetric",
        )
        finest_detail = coefficients[-1]

        universal_th = calculate_universal_threshold(
            finest_detail, len(noisy_ecg)
        )
        minimax_th = calculate_minimax_threshold(
            finest_detail, len(noisy_ecg)
        )
        heursure_th = calculate_heursure_threshold(
            finest_detail, len(noisy_ecg)
        )

        methods = {
            "Universal": universal_th,
            "Minimax": minimax_th,
            "Heursure": heursure_th,
        }

        results = {}

        for method_name, threshold in methods.items():
            denoised = wavelet_denoise_with_fixed_threshold(
                noisy_signal=noisy_ecg,
                wavelet_name=wavelet_name,
                decomposition_level=decomposition_level,
                threshold=threshold,
                threshold_mode=threshold_mode,
            )

            output_snr = calculate_snr(clean_ecg, denoised)
            output_rmse = calculate_rmse(clean_ecg, denoised)
            snr_improvement = output_snr - actual_input_snr

            results[method_name] = {
                "snr": output_snr,
                "rmse": output_rmse,
                "snr_imp": snr_improvement,
            }

        bayes_denoised, bayes_thresholds = wavelet_denoise_bayes(
            noisy_signal=noisy_ecg,
            wavelet_name=wavelet_name,
            decomposition_level=decomposition_level,
            threshold_mode=threshold_mode,
        )
        bayes_snr = calculate_snr(clean_ecg, bayes_denoised)
        bayes_rmse = calculate_rmse(clean_ecg, bayes_denoised)
        results["Bayes"] = {
            "snr": bayes_snr,
            "rmse": bayes_rmse,
            "snr_imp": bayes_snr - actual_input_snr,
        }

        print(f"\n  Running ACF method for {noise_name}...")
        acf_denoised, acf_th, acf_nzopp, acf_iter = wavelet_denoise_acf(
            noisy_signal=noisy_ecg,
            wavelet_name=wavelet_name,
            decomposition_level=decomposition_level,
            threshold_mode=threshold_mode,
            verbose=False,
        )
        acf_snr = calculate_snr(clean_ecg, acf_denoised)
        acf_rmse = calculate_rmse(clean_ecg, acf_denoised)
        results["ACF (Proposed)"] = {
            "snr": acf_snr,
            "rmse": acf_rmse,
            "snr_imp": acf_snr - actual_input_snr,
        }

        print(f"  ACF threshold: {acf_th:.6f}, NZOPP: {acf_nzopp:.6f}")
        print(f"  ACF SNR: {acf_snr:.4f} dB, RMSE: {acf_rmse:.6f}")


        print(f"\n  {'Method':<20} {'SNR (dB)':>10} {'RMSE':>10} {'SNR Imp':>10}")
        print(f"  {'-' * 50}")

        for method_name, result in results.items():
            print(
                f"  {method_name:<20} "
                f"{result['snr']:>10.4f} "
                f"{result['rmse']:>10.6f} "
                f"{result['snr_imp']:>10.4f}"
            )

        all_results[noise_name] = results

        # Save denoised signals
        np.save(
            SIGNAL_DIR / f"record109_{noise_safe_name}_acf_denoised.npy",
            acf_denoised,
        )

    print(f"\n{'=' * 70}")
    print("SUMMARY: SNR IMPROVEMENT COMPARISON")
    print(f"{'=' * 70}")

    for noise_name, results in all_results.items():
        print(f"\n{noise_name}:")
        print(f"  {'Method':<20} {'SNR Improvement (dB)':>22} {'RMSE':>12}")
        print(f"  {'-' * 54}")

        for method_name, result in results.items():
            print(
                f"  {method_name:<20} "
                f"{result['snr_imp']:>22.4f} "
                f"{result['rmse']:>12.6f}"
            )

    summary_path = RESULT_DIR / "phase6_real_noise_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Real Noise Experiments - Summary\n")
        f.write("=" * 70 + "\n")

        for noise_name, results in all_results.items():
            f.write(f"\n{noise_name}:\n")
            f.write(
                f"  {'Method':<20} "
                f"{'SNR Improvement (dB)':>22} "
                f"{'RMSE':>12}\n"
            )
            f.write(f"  {'-' * 54}\n")

            for method_name, result in results.items():
                f.write(
                    f"  {method_name:<20} "
                    f"{result['snr_imp']:>22.4f} "
                    f"{result['rmse']:>12.6f}\n"
                )

    print(f"\n[SAVED SUMMARY] {summary_path}")

# ============================================================
# 10. Main
# ============================================================
def main() -> None:
    print_section("ACF WAVELET ECG DENOISING PROJECT")

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Dataset dir  : {DATASET_DIR}")
    print(f"Output dir   : {OUTPUT_DIR}")

    # Phase 1
    validate_datasets()
    run_initial_data_test()

    # Phase 2
    run_awgn_generation_test()

    # Phase 3
    run_wavelet_pipeline_test()

    # Phase 4
    run_classical_methods_comparison()

    # Phase 5
    run_acf_method_comparison()

    # Phase 6
    run_real_noise_experiment()

    print_section("PROJECT COMPLETED")
    print("All phases executed successfully.")
    print("Results are saved in the outputs/ directory.")


if __name__ == "__main__":
    main()
