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
import tempfile

st.set_page_config(page_title="Sedline EDF DSA 分析", layout="wide")
st.title("Sedline EDF DSA 動態頻譜分析 (DSA)")

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
        return datetime.now()

def multitaper_dsa_sliding_mne(eeg, fs, win_sec=3, step_sec=1, fmin=1, fmax=40):
    n_win = int(win_sec * fs)
    n_step = int(step_sec * fs)
    n_total = len(eeg)
    starts = np.arange(0, n_total - n_win + 1, n_step)
    if len(starts) == 0:
        st.warning("資料太短，無法計算 DSA")
        return None, None, None
    power_all = []
    freqs_all = None
    for start in starts:
        segment = eeg[start:start + n_win]
        try:
            psd, freqs = psd_array_multitaper(
                segment, sfreq=fs, fmin=fmin, fmax=fmax,
                adaptive=False, normalization='full', verbose=False)
            if freqs_all is None:
                freqs_all = freqs
            power_all.append(psd)
        except Exception as e:
            st.warning(f"MNE multitaper 計算錯誤，位置 {start}: {e}")
            continue
    if len(power_all) == 0:
        return None, None, None
    power_all = np.stack(power_all, axis=1)
    t_centers = (starts + n_win // 2) / fs
    return power_all, freqs_all, t_centers

if uploaded_files:
    # 按時間排序
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
            # 用暫存檔寫入並讀取
            with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp_file:
                tmp_file.write(fobj.read())
                tmp_filename = tmp_file.name
            f = pyedflib.EdfReader(tmp_filename)
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
            st.warning("不同檔案頻率軸不一致，暫不處理插值")

        time_offset_seconds = (file_datetime - dataset_start_time).total_seconds()
        actual_times_relative_to_dataset_start = [dataset_start_time + timedelta(seconds=float(sec) + time_offset_seconds) for sec in t_centers]

        powers.append(power)
        all_times.extend(actual_times_relative_to_dataset_start)

    if powers:
        first_freq_dim = powers[0].shape[0]
        if not all(p.shape[0] == first_freq_dim for p in powers):
            st.error("頻率維度不一致，無法合併")
            concat_power = None
        else:
            concat_power = np.concatenate(powers, axis=1)

        if concat_power is not None and all_times:
            vmin = st.slider("Power dB 最小值", -60, 0, -40, 5)
            vmax = st.slider("Power dB 最大值", -20, 30, 10, 5)
            fmin_plot = st.slider("頻率最低值 (Hz)", int(freqs[0]), int(freqs[-1]), 1)
            fmax_plot = st.slider("頻率最高值 (Hz)", int(freqs[0]), int(freqs[-1]), 40)
            cmap = st.selectbox("色彩映射", ['jet', 'viridis', 'plasma', 'inferno', 'magma', 'cividis'])

            extent = [mdates.date2num(all_times[0]), mdates.date2num(all_times[-1]), freqs[0], freqs[-1]]
            fig, ax = plt.subplots(figsize=(12, 4))
            im = ax.imshow(10 * np.log10(concat_power), aspect='auto', origin='lower', extent=extent,
                           vmin=vmin, vmax=vmax, cmap=cmap)
            ax.set_ylabel('Frequency (Hz)')
            ax.set_xlabel('Time')
            ax.set_title('Concatenated DSA')
            ax.set_ylim([fmin_plot, fmax_plot])
            ax.xaxis_date()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.xticks(rotation=30)
            fig.colorbar(im, ax=ax, label='Power (dB)')
            st.pyplot(fig)
        else:
            st.warning("無法繪製 DSA，請確認檔案與分析結果")
    else:
        st.warning("沒有有效的分析數據")

else:
    st.info("請先上傳至少一個 EDF 檔案以開始分析")
