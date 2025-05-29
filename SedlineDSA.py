import streamlit as st
import pyedflib
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from mne.time_frequency import psd_array_multitaper
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import re

st.title("Sedline EDF DSA 分析（Streamlit 版本）")

uploaded_files = st.file_uploader("請上傳一個或多個 EDF 檔案", type="edf", accept_multiple_files=True)

def extract_datetime_from_filename(fname):
    match = re.match(r'EEG_(\d{6})_(\d{6})\.edf', fname)
    if match:
        date_str = match.group(1)
        time_str = match.group(2)
        dt = datetime.strptime(date_str + time_str, "%y%m%d%H%M%S")
        if dt > datetime.now() + timedelta(days=365):
            dt = dt - timedelta(days=365*100)
        return dt
    else:
        try:
            f = pyedflib.EdfReader(fname)
            start_datetime = f.getStartdatetime()
            f._close()
            return start_datetime
        except:
            return datetime.now()

def multitaper_dsa_sliding_mne(eeg, fs, win_sec=3, step_sec=1, fmin=1, fmax=40):
    n_win = int(win_sec * fs)
    n_step = int(step_sec * fs)
    n_total = len(eeg)
    starts = np.arange(0, n_total - n_win + 1, n_step)
    if len(starts) == 0:
        return None, None, None
    power_all = []
    freqs_all = None
    for start in starts:
        segment = eeg[start:start + n_win]
        try:
            psd, freqs = psd_array_multitaper(segment, sfreq=fs, fmin=fmin, fmax=fmax,
                                              adaptive=False, normalization='full', verbose=False)
            if freqs_all is None:
                freqs_all = freqs
            power_all.append(psd)
        except:
            continue
    if len(power_all) == 0:
        return None, None, None
    power_all = np.stack(power_all, axis=1)
    t_centers = (starts + n_win // 2) / fs
    return power_all, freqs_all, t_centers

if uploaded_files:
    # 依時間排序
    edf_files = sorted(uploaded_files, key=lambda f: extract_datetime_from_filename(f.name))
    powers = []
    freqs = None
    all_times = []
    dataset_start_time = None

    for fobj in edf_files:
        fname = fobj.name
        st.write(f"處理檔案: {fname}")
        file_datetime = extract_datetime_from_filename(fname)

        try:
            f = pyedflib.EdfReader(fobj)
            eeg = f.readSignal(0)
            fs = f.getSampleFrequency(0)
            f._close()
        except Exception as e:
            st.error(f"讀取 EDF 失敗: {e}")
            continue

        if dataset_start_time is None:
            dataset_start_time = file_datetime

        eeg = eeg.astype(np.float32)
        mean = np.mean(eeg)
        std = np.std(eeg)
        if std == 0:
            eeg_clean = eeg.copy()
        else:
            mask = (eeg > mean - 3*std) & (eeg < mean + 3*std)
            bad_idx = np.where(~mask)[0]
            if len(bad_idx) > 0:
                good_idx = np.where(mask)[0]
                if len(good_idx) > 0:
                    interp_func = interp1d(good_idx, eeg[good_idx], bounds_error=False, fill_value="extrapolate")
                    eeg_clean = eeg.copy()
                    eeg_clean[bad_idx] = interp_func(bad_idx)
                else:
                    eeg_clean = eeg.copy()
            else:
                eeg_clean = eeg.copy()

        try:
            b_hp, a_hp = butter(3, 0.2 / (fs/2), btype='high')
            b_lp, a_lp = butter(3, 30 / (fs/2), btype='low')
            filtered = filtfilt(b_hp, a_hp, eeg_clean)
            filtered = filtfilt(b_lp, a_lp, filtered)
        except Exception as e:
            st.error(f"濾波失敗: {e}")
            continue

        power, cur_freqs, t_centers = multitaper_dsa_sliding_mne(filtered, fs)
        if power is None:
            st.warning("DSA 計算失敗")
            continue

        if freqs is None:
            freqs = cur_freqs
        elif not np.array_equal(freqs, cur_freqs):
            # 插值處理省略，需時可補充
            pass

        time_offset_seconds = (file_datetime - dataset_start_time).total_seconds()
        actual_times_relative_to_dataset_start = [dataset_start_time + timedelta(seconds=float(sec) + time_offset_seconds) for sec in t_centers]

        powers.append(power)
        all_times.extend(actual_times_relative_to_dataset_start)

    if powers:
        concat_power = np.concatenate(powers, axis=1)
        # 顯示圖形與互動元件
        vmin = st.slider('最小 Power (dB)', -60, 0, -40, 5)
        vmax = st.slider('最大 Power (dB)', -20, 30, 10, 5)
        fmin = st.slider('最低頻率 (Hz)', int(freqs[0]), int(freqs[-1]), 1)
        fmax = st.slider('最高頻率 (Hz)', int(freqs[0]), int(freqs[-1]), 40)
        cmap = st.selectbox('色彩映射', ['jet', 'viridis', 'plasma', 'inferno', 'magma', 'cividis'])

        extent = [mdates.date2num(all_times[0]), mdates.date2num(all_times[-1]), freqs[0], freqs[-1]]
        fig, ax = plt.subplots(figsize=(12, 4))
        im = ax.imshow(10 * np.log10(concat_power), aspect='auto', origin='lower', extent=extent,
                       vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time')
        ax.set_title('Concatenated DSA')
        ax.set_ylim([fmin, fmax])
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.xticks(rotation=30)
        fig.colorbar(im, ax=ax, label='Power (dB)')
        st.pyplot(fig)

    else:
        st.warning("沒有有效的分析數據。")

else:
    st.info("請上傳至少一個 EDF 檔案")
