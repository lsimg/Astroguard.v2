"""
AstroGuard v3 -- Universal Autonomous Edge AI for Satellite Prognostics

Core innovation: the system automatically adapts to ANY telemetry dataset.
It detects the data format, extracts features, builds its own neural
architecture, trains itself, and predicts remaining useful life --
all without manual configuration.

Supported formats (auto-detected):
  - NASA IMS: folder of timestamped tab-separated vibration files
  - NASA SMAP/MSL: train/test .npy arrays with labeled_anomalies.csv
  - Generic CSV/TSV: any tabular time-series data
  - Generic NumPy: folder of .npy arrays
"""

import os, time, math, warnings, glob, ast as _ast
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class DatasetProfile:
    name: str
    format_type: str
    train_data: np.ndarray
    test_data: np.ndarray
    feature_names: list
    anomaly_labels: object
    has_vibration: bool
    sample_rate: int
    time_labels: list
    channel_id: str
    raw_test_signals: object


class UniversalLoader:

    @staticmethod
    def detect_format(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")
        contents = os.listdir(path)
        if "train" in contents and "test" in contents:
            train_dir = os.path.join(path, "train")
            if any(f.endswith(".npy") for f in os.listdir(train_dir)):
                return "smap_msl"
        text_files = [f for f in contents if not f.startswith(".") and "." in f]
        if text_files:
            parts = text_files[0].split(".")
            if len(parts) >= 6:
                try:
                    int(parts[0]); return "ims"
                except ValueError:
                    pass
        if any(f.endswith(".npy") for f in contents):
            return "npy_generic"
        if any(f.endswith(".csv") for f in contents):
            return "csv_generic"
        tsv_candidates = [f for f in contents if not f.startswith(".")]
        if tsv_candidates:
            try:
                with open(os.path.join(path, tsv_candidates[0])) as fh:
                    if "\t" in fh.readline(): return "ims"
            except Exception:
                pass
        raise ValueError(f"Cannot detect format in '{path}'")

    @staticmethod
    def load(path, channel="auto"):
        fmt = UniversalLoader.detect_format(path)
        loaders = {"ims": UniversalLoader._load_ims, "smap_msl": UniversalLoader._load_smap,
                    "npy_generic": UniversalLoader._load_npy, "csv_generic": UniversalLoader._load_csv}
        if fmt == "smap_msl":
            return loaders[fmt](path, channel)
        return loaders[fmt](path)

    @staticmethod
    def _load_ims(path):
        files = sorted([f for f in os.listdir(path) if not f.startswith(".")])
        if len(files) < 100:
            raise ValueError(f"IMS needs 100+ files, found {len(files)}")
        split = int(len(files) * 0.6)
        def extract(flist):
            feats, sigs = [], []
            for fn in flist:
                try:
                    df = pd.read_csv(os.path.join(path, fn), sep="\t", header=None)
                    row = []
                    for c in df.columns:
                        s = df[c].values.astype(np.float64)
                        row.extend([np.sqrt(np.mean(s**2)), np.max(np.abs(s)), float(pd.Series(s).kurtosis()), np.std(s)])
                    feats.append(row); sigs.append(df[0].values.astype(np.float64))
                except Exception: continue
            return np.array(feats, dtype=np.float64), sigs
        train_f, _ = extract(files[:split])
        test_f, test_s = extract(files[split:])
        nb = train_f.shape[1] // 4 if len(train_f) > 0 else 1
        names = [f"B{b+1}_{f}" for b in range(nb) for f in ["RMS","Peak","Kurt","Std"]]
        tlabels = []
        for fn in files[split:]:
            try:
                p = fn.split("."); dt = datetime.strptime(".".join(p[:6]), "%Y.%m.%d.%H.%M.%S")
                tlabels.append(dt.strftime("%Y-%m-%d %H:%M:%S UTC"))
            except Exception: tlabels.append(fn)
        return DatasetProfile("NASA IMS Bearing Dataset", "ims", train_f, test_f, names, None, True, 20000, tlabels, "all_bearings", test_s)

    @staticmethod
    def _load_smap(path, channel="auto"):
        train_dir, test_dir = os.path.join(path, "train"), os.path.join(path, "test")
        channels = sorted([f[:-4] for f in os.listdir(train_dir) if f.endswith(".npy")])
        if not channels: raise ValueError("No .npy in train/")
        labels_path = os.path.join(path, "labeled_anomalies.csv")
        labels_df = pd.read_csv(labels_path) if os.path.exists(labels_path) else None
        if channel == "auto": channel = channels[0]
        if channel not in channels: raise ValueError(f"Channel '{channel}' not found")
        train_raw = np.load(os.path.join(train_dir, f"{channel}.npy"))
        test_raw = np.load(os.path.join(test_dir, f"{channel}.npy"))
        window = 50
        def featurize(data):
            if data.ndim == 1: data = data.reshape(-1, 1)
            nc = data.shape[1]; rows = []
            for i in range(0, len(data) - window + 1, window):
                seg = data[i:i+window]; row = []
                for c in range(nc):
                    col = seg[:, c]
                    row.extend([np.mean(col), np.std(col), np.max(col)-np.min(col),
                                float(pd.Series(col).kurtosis()) if np.std(col) > 1e-10 else 0.0])
                rows.append(row)
            return np.array(rows, dtype=np.float64)
        train_feat, test_feat = featurize(train_raw), featurize(test_raw)
        nc = train_raw.shape[1] if train_raw.ndim > 1 else 1
        names = [f"Ch{c}_{f}" for c in range(nc) for f in ["Mean","Std","Range","Kurt"]]
        anom = None
        if labels_df is not None:
            row = labels_df[labels_df["chan_id"] == channel]
            if len(row) > 0:
                seqs = _ast.literal_eval(row.iloc[0]["anomaly_sequences"])
                la = np.zeros(len(test_raw), dtype=np.float64)
                for s in seqs: la[s[0]:s[1]] = 1.0
                anom = np.array([np.max(la[i:i+window]) for i in range(0, len(la)-window+1, window)])
        sc = "SMAP/MSL"
        if labels_df is not None:
            r = labels_df[labels_df["chan_id"] == channel]
            if len(r) > 0: sc = r.iloc[0]["spacecraft"]
        return DatasetProfile(f"NASA {sc} ({channel})", "smap_msl", train_feat, test_feat, names, anom, False, 0,
                              [f"Step {i}" for i in range(len(test_feat))], channel, None)

    @staticmethod
    def _load_csv(path):
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
        parts = []
        for f in files:
            try:
                df = pd.read_csv(f); nc = df.select_dtypes(include=[np.number]).columns
                if len(nc) > 0: parts.append(df[nc].values)
            except Exception: continue
        if not parts: raise ValueError("No numeric CSV data")
        data = np.vstack(parts).astype(np.float64)
        sp = int(len(data) * 0.6)
        return DatasetProfile("Generic CSV", "csv_generic", data[:sp], data[sp:],
                              [f"F{i}" for i in range(data.shape[1])], None, False, 0,
                              [f"Row {i}" for i in range(len(data)-sp)], "all", None)

    @staticmethod
    def _load_npy(path):
        files = sorted(glob.glob(os.path.join(path, "*.npy")))
        parts = [np.load(f) if np.load(f).ndim > 1 else np.load(f).reshape(-1,1) for f in files]
        data = np.vstack(parts).astype(np.float64)
        sp = int(len(data) * 0.6)
        return DatasetProfile("Generic NumPy", "npy_generic", data[:sp], data[sp:],
                              [f"D{i}" for i in range(data.shape[1])], None, False, 0,
                              [f"Step {i}" for i in range(len(data)-sp)], "all", None)

    @staticmethod
    def list_channels(path):
        try:
            if UniversalLoader.detect_format(path) == "smap_msl":
                return sorted([f[:-4] for f in os.listdir(os.path.join(path, "train")) if f.endswith(".npy")])
        except Exception: pass
        return []


class Autoencoder(nn.Module):
    def __init__(self, dim, layers, act="tanh"):
        super().__init__()
        fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "elu": nn.ELU, "selu": nn.SELU}[act]
        enc = []
        p = dim
        for s in layers: enc += [nn.Linear(p, s), fn()]; p = s
        self.encoder = nn.Sequential(*enc)
        dec = []
        mir = list(reversed(layers[:-1])) + [dim]
        p = layers[-1]
        for s in mir: dec += [nn.Linear(p, s), fn()]; p = s
        dec[-1] = nn.Sigmoid()
        self.decoder = nn.Sequential(*dec)
    def forward(self, x): return self.decoder(self.encoder(x))


@dataclass
class ArchCandidate:
    layers: list; activation: str; lr: float
    val_loss: float = float("inf"); train_time: float = 0.0


class AutoBuilder:
    ACTS = ["tanh", "relu", "elu"]
    LRS = [1e-3, 5e-4, 1e-4]

    def __init__(self, dim, data, cb=None):
        self.dim, self.cb = dim, cb
        self.templates = self._make_templates(dim)
        sp = int(len(data) * 0.85)
        self.tr = torch.FloatTensor(data[:sp]).to(DEVICE)
        self.va = torch.FloatTensor(data[sp:]).to(DEVICE)
        self.ds = TensorDataset(self.tr, self.tr)

    @staticmethod
    def _make_templates(d):
        b = max(16, d)
        raw = [[b*2,b],[b*4,b*2,b],[b*8,b*4,b*2],[b*4,b*2,b,max(8,b//2)],[b*2,b,max(8,b//2)],[b*8,b*4,b*2,b]]
        return [[max(8, min(512, x)) for x in t] for t in raw]

    def _eval(self, c):
        m = Autoencoder(self.dim, c.layers, c.activation).to(DEVICE)
        opt = optim.Adam(m.parameters(), lr=c.lr); fn = nn.MSELoss()
        ld = DataLoader(self.ds, batch_size=min(64, len(self.ds)), shuffle=True)
        best, stag, t0 = float("inf"), 0, time.time()
        for _ in range(60):
            m.train()
            for x, y in ld: opt.zero_grad(); l = fn(m(x), y); l.backward(); opt.step()
            m.eval()
            with torch.no_grad(): vl = fn(m(self.va), self.va).item()
            if vl < best: best, stag = vl, 0
            else:
                stag += 1
                if stag >= 8: break
        c.val_loss, c.train_time = best, time.time() - t0
        return c

    def search(self, max_n=54):
        pool = [ArchCandidate(l, a, r) for l in self.templates for a in self.ACTS for r in self.LRS]
        if len(pool) > max_n:
            np.random.seed(42); idx = np.random.choice(len(pool), max_n, replace=False)
            pool = [pool[i] for i in idx]
        res = []
        for i, c in enumerate(pool):
            if self.cb: self.cb(i, len(pool), c)
            res.append(self._eval(c))
        res.sort(key=lambda e: e.val_loss)
        return res[0], res


def train_final(dim, arch, data, epochs=200, cb=None):
    m = Autoencoder(dim, arch.layers, arch.activation).to(DEVICE)
    opt = optim.Adam(m.parameters(), lr=arch.lr)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
    fn = nn.MSELoss(); dt = torch.FloatTensor(data).to(DEVICE)
    ld = DataLoader(TensorDataset(dt, dt), batch_size=min(64, len(data)), shuffle=True)
    best_l, best_w = float("inf"), None
    for ep in range(epochs):
        m.train(); tl = 0
        for x, y in ld: opt.zero_grad(); l = fn(m(x), y); l.backward(); opt.step(); tl += l.item()
        avg = tl / len(ld); sch.step(avg)
        if avg < best_l: best_l = avg; best_w = {k: v.clone() for k, v in m.state_dict().items()}
        if cb and ep % 10 == 0: cb(ep, epochs, avg)
    if best_w: m.load_state_dict(best_w)
    m.eval()
    with torch.no_grad(): errs = torch.mean((dt - m(dt))**2, dim=1).cpu().numpy()
    return m, float(np.mean(errs) + 3.0 * np.std(errs))


class RULPredictor:
    def __init__(self, thresh, minpts=15):
        self.thresh, self.minpts, self._p, self._ok = thresh, minpts, None, False
    @staticmethod
    def _curve(t, a, b, c): return a * np.exp(b * t) + c
    def update(self, scores):
        if len(scores) < self.minpts: return None
        try:
            p, _ = curve_fit(self._curve, np.arange(len(scores), dtype=np.float64),
                             np.array(scores, dtype=np.float64), p0=[1e-6, 0.01, min(scores)],
                             maxfev=5000, bounds=([0,0,-np.inf],[np.inf,1,np.inf]))
            self._p, self._ok = p, True
            a, b, c = p
            if a <= 0 or b <= 0 or self.thresh <= c: return None
            arg = (self.thresh - c) / a
            if arg <= 0: return None
            rem = math.log(arg) / b - len(scores)
            return max(0.0, rem * 10 / 60)
        except Exception: self._ok = False; return None
    def get_curve(self, n, ahead=50):
        if not self._ok or self._p is None: return None
        t = np.arange(0, n + ahead, dtype=np.float64)
        return t, self._curve(t, *self._p)


def spectral_analysis(sig, sr, bpfo=166.0, bpfi=236.0, tol=8.0):
    n = len(sig); sp = fft(sig); fr = fftfreq(n, 1/sr); pos = fr >= 0
    freqs, amps = fr[pos], 2/n * np.abs(sp[pos])
    result = {"freqs": freqs, "amps": amps, "bpfo_det": False, "bpfi_det": False,
              "bpfo_amp": 0, "bpfi_amp": 0, "dom_freq": 0, "confirmed": False}
    if len(amps) == 0: return result
    result["dom_freq"] = float(freqs[np.argmax(amps[1:])+1])
    med = float(np.median(amps[1:])); th = med * 5
    for target, ka, kd in [(bpfo, "bpfo_amp", "bpfo_det"), (bpfi, "bpfi_amp", "bpfi_det")]:
        band = (freqs >= target-tol) & (freqs <= target+tol)
        if np.any(band):
            result[ka] = float(np.max(amps[band]))
            if result[ka] > th: result[kd] = True
    result["confirmed"] = result["bpfo_det"] or result["bpfi_det"]
    return result


def classify(score, th):
    if score < th: return "status-ok", "NOMINAL"
    elif score < th * 2: return "status-warn", "ADVISORY"
    else: return "status-crit", "CRITICAL"

def fmt_rul(h):
    if h is None: return "CALCULATING...", "#7b93b8"
    if h > 240: return "> 10 DAYS", "#00e68a"
    if h > 24: return f"{h:.1f} HOURS", "#ffbe0b"
    if h > 0: return f"{h:.1f} HOURS", "#ff3b5c"
    return "IMMINENT", "#ff3b5c"


def plot_health(data, th, crit, pred):
    fig, ax = plt.subplots(figsize=(10,3), dpi=130); fig.patch.set_facecolor("#03070f"); ax.set_facecolor("#070e1c")
    ax.plot(data, color="#00e68a", lw=1.8, label="Anomaly Score", zorder=3)
    ax.axhline(th, color="#ff3b5c", ls="--", lw=1.2, label="Alert", zorder=2)
    ax.axhline(crit, color="#ffbe0b", ls=":", lw=1.0, label="Critical", zorder=2)
    c = pred.get_curve(len(data), 40)
    if c: ax.plot(c[0], c[1], color="#a78bfa", ls="--", lw=0.9, alpha=0.7, label="Exp. Fit")
    ax.set_ylim(0, max(max(data), th, crit) * 1.15 or 1)
    ax.set_xlabel("Step", color="#7b93b8", fontsize=8); ax.set_ylabel("Score", color="#7b93b8", fontsize=8)
    ax.tick_params(colors="#7b93b8", labelsize=8)
    for s in ax.spines.values(): s.set_color("#1a2d4d")
    ax.grid(color="#111e36", ls="--", lw=0.5, alpha=0.6)
    ax.legend(loc="upper left", frameon=False, fontsize=7, labelcolor="#dce6f5"); return fig

def plot_fft(r, slen, sr):
    fig, ax = plt.subplots(figsize=(10,2.5), dpi=130); fig.patch.set_facecolor("#03070f"); ax.set_facecolor("#070e1c")
    ax.plot(r["freqs"], r["amps"], color="#38bdf8", lw=0.7)
    ax.axvspan(158, 174, alpha=0.15, color="#ff3b5c", label="BPFO ~166 Hz")
    ax.axvspan(228, 244, alpha=0.15, color="#ffbe0b", label="BPFI ~236 Hz")
    fl = min(1000, float(r["freqs"][-1])) if len(r["freqs"]) else 1000; ax.set_xlim(0, fl)
    ci = int(fl / (sr / slen) + 1); ym = max(0.05, float(np.max(r["amps"][:ci])) * 1.2) if len(r["amps"]) > 0 else 0.2
    ax.set_ylim(0, ym); ax.set_xlabel("Frequency (Hz)", color="#7b93b8", fontsize=8); ax.set_ylabel("Amplitude", color="#7b93b8", fontsize=8)
    ax.tick_params(colors="#7b93b8", labelsize=7)
    for s in ax.spines.values(): s.set_color("#1a2d4d")
    ax.grid(color="#111e36", ls="--", lw=0.5, alpha=0.5); ax.legend(loc="upper right", frameon=False, fontsize=7, labelcolor="#dce6f5"); return fig

def plot_gt(data, labels, th, crit):
    fig, ax = plt.subplots(figsize=(10,2.5), dpi=130); fig.patch.set_facecolor("#03070f"); ax.set_facecolor("#070e1c")
    ax.plot(data, color="#00e68a", lw=1.2, label="Anomaly Score", zorder=3)
    ax.axhline(th, color="#ff3b5c", ls="--", lw=1.0, label="Alert", zorder=2)
    if labels is not None and len(labels) == len(data):
        mask = np.array(labels) > 0
        if np.any(mask):
            groups = np.split(np.where(mask)[0], np.where(np.diff(np.where(mask)[0]) != 1)[0] + 1)
            for g in groups:
                if len(g): ax.axvspan(g[0], g[-1], alpha=0.2, color="#ff3b5c", zorder=1)
            ax.plot([], [], color="#ff3b5c", alpha=0.3, lw=6, label="Ground Truth")
    ax.set_ylim(0, max(max(data), th, crit) * 1.15 or 1)
    ax.set_xlabel("Step", color="#7b93b8", fontsize=8); ax.set_ylabel("Score", color="#7b93b8", fontsize=8)
    ax.tick_params(colors="#7b93b8", labelsize=8)
    for s in ax.spines.values(): s.set_color("#1a2d4d")
    ax.grid(color="#111e36", ls="--", lw=0.5, alpha=0.6); ax.legend(loc="upper left", frameon=False, fontsize=7, labelcolor="#dce6f5"); return fig


st.set_page_config(page_title="AstroGuard v3", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Outfit:wght@300;500;700;900&display=swap');
:root{--ag-bg-1:#03070f;--ag-bg-2:#070e1c;--ag-text:#dce6f5;--ag-muted:#7b93b8;--ag-border:#1a2d4d;--ag-accent:#00d4ff;--ag-ok:#00e68a;--ag-warn:#ffbe0b;--ag-crit:#ff3b5c;--ag-purple:#a78bfa}
.stApp{background:radial-gradient(ellipse 900px 500px at 5% -5%,rgba(0,100,200,.12) 0%,transparent 70%),radial-gradient(ellipse 600px 400px at 95% 90%,rgba(100,0,200,.06) 0%,transparent 70%),linear-gradient(180deg,var(--ag-bg-1),var(--ag-bg-2));color:var(--ag-text);font-family:'Outfit',sans-serif}
[data-testid="stHeader"]{background:transparent!important}[data-testid="stDecoration"]{display:none!important}
[data-testid="stToolbar"],[data-testid="stStatusWidget"],#MainMenu,button[title="View options"],button[title="Manage app"],button[title="Settings"]{display:none!important}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#060d1a,#040a14);border-right:1px solid var(--ag-border)}
[data-testid="stSidebar"] *{color:var(--ag-text)!important}
h1,h2,h3,p,label,span{color:var(--ag-text)!important}
h1{font-family:'Outfit',sans-serif!important;font-weight:900!important;letter-spacing:-.02em!important}
.metric-card{background:linear-gradient(135deg,rgba(12,22,40,.95),rgba(7,14,28,.95));border:1px solid var(--ag-border);border-left:5px solid var(--ag-muted);border-radius:8px;padding:14px 16px;margin-bottom:8px;font-family:'JetBrains Mono',monospace;font-size:13px;line-height:1.6}
.time-display{font-family:'JetBrains Mono',monospace;font-size:28px;color:var(--ag-accent);background:rgba(0,0,0,.4);border:1px solid var(--ag-border);border-radius:8px;padding:10px;text-align:center}
.rul-display{font-family:'Outfit',sans-serif;font-size:44px;font-weight:900;text-align:center;border:1px solid var(--ag-border);border-radius:8px;padding:12px;background:rgba(0,0,0,.3)}
.arch-card{background:rgba(10,19,34,.9);border:1px solid var(--ag-border);border-radius:8px;padding:12px;margin:6px 0;font-family:'JetBrains Mono',monospace;font-size:12px}
.format-badge{display:inline-block;padding:4px 12px;border-radius:4px;font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:700;background:rgba(0,212,255,.15);color:var(--ag-accent);border:1px solid rgba(0,212,255,.3)}
.status-ok{border-left-color:var(--ag-ok)!important}.status-warn{border-left-color:var(--ag-warn)!important}.status-crit{border-left-color:var(--ag-crit)!important}
button[kind]{background:linear-gradient(135deg,#0c1e3a,#142850)!important;border:1px solid var(--ag-border)!important;color:var(--ag-text)!important;font-weight:600!important}
</style>""", unsafe_allow_html=True)

st.title("AstroGuard v3: Universal AI Prognostics")
st.caption("Auto-adapts to any telemetry format -- IMS, SMAP, MSL, CSV, NumPy")

st.sidebar.header("Control Panel")
if "data_path" not in st.session_state: st.session_state["data_path"] = "./IMS"
st.sidebar.text_input("Data Source:", key="data_path")
st.sidebar.caption("Any folder: IMS, SMAP/MSL (with train/test), CSV, or NumPy")

det_fmt, det_ch = "...", []
try: det_fmt = UniversalLoader.detect_format(st.session_state["data_path"]); det_ch = UniversalLoader.list_channels(st.session_state["data_path"])
except Exception: pass
st.sidebar.markdown(f"Detected: <span class='format-badge'>{det_fmt}</span>", unsafe_allow_html=True)

ch_sel = "auto"
if det_ch: ch_sel = st.sidebar.selectbox("Telemetry Channel:", ["auto"] + det_ch, help="For SMAP/MSL: pick channel")
speed = st.sidebar.slider("Replay Speed (ms)", 10, 500, 50)
win = st.sidebar.slider("Smoothing Window", 5, 50, 20)
st.sidebar.divider(); st.sidebar.subheader("AutoBuilder Settings")
max_cand = st.sidebar.select_slider("Search Breadth", options=[6,12,18,27,54], value=18)
fin_ep = st.sidebar.slider("Final Training Epochs", 50, 500, 200)
st.sidebar.divider()
go = st.sidebar.button("INITIATE MISSION", use_container_width=True)

if go:
    dp = st.session_state["data_path"]
    p0 = st.status("Phase 0: Detecting data format...", expanded=True)
    with p0:
        try: ds = UniversalLoader.load(dp, channel=ch_sel)
        except Exception as e: st.error(str(e)); st.stop()
        st.write(f"**Dataset:** {ds.name}"); st.write(f"**Format:** {ds.format_type}"); st.write(f"**Channel:** {ds.channel_id}")
        st.write(f"**Train:** {len(ds.train_data)} samples, **Test:** {len(ds.test_data)} samples")
        st.write(f"**Features:** {len(ds.feature_names)} -- {', '.join(ds.feature_names[:8])}{'...' if len(ds.feature_names)>8 else ''}")
        st.write(f"**Vibration:** {'Yes (FFT enabled)' if ds.has_vibration else 'No'}")
        if ds.anomaly_labels is not None: st.write(f"**Ground truth:** {int(np.sum(ds.anomaly_labels>0))} anomalous windows")
    p0.update(label=f"Phase 0: {ds.format_type} loaded", state="complete", expanded=False)

    if len(ds.train_data) < 30: st.error("Need 30+ training samples"); st.stop()
    sc = MinMaxScaler(); norm_tr = sc.fit_transform(ds.train_data); idim = norm_tr.shape[1]

    p1 = st.status("Phase 1: AutoBuilder -- searching architecture...", expanded=True)
    with p1:
        pb = st.progress(0); pt = st.empty()
        def scb(i, t, c): pb.progress((i+1)/t); pt.write(f"Testing **{i+1}/{t}**: layers={c.layers}, act={c.activation}, lr={c.lr}")
        bld = AutoBuilder(idim, norm_tr, cb=scb); best, allr = bld.search(max_cand)
        pb.progress(1.0); pt.empty()
        enc = " -> ".join(map(str, best.layers)); dec = " -> ".join(map(str, reversed(best.layers[:-1])))
        st.write("### Best Architecture Found")
        st.markdown(f'<div class="arch-card"><b>Layers:</b> {idim} -> {enc} -> {dec} -> {idim}<br><b>Activation:</b> {best.activation}<br><b>LR:</b> {best.lr}<br><b>Val MSE:</b> {best.val_loss:.6f}<br><b>Time:</b> {best.train_time:.1f}s</div>', unsafe_allow_html=True)
        st.write("**Top 5:**")
        for i, r in enumerate(allr[:5]): st.caption(f"{'  ' if i==0 else f'#{i+1}'}  {r.layers}, {r.activation}, lr={r.lr}, loss={r.val_loss:.6f}")
    p1.update(label="Phase 1: Architecture selected", state="complete", expanded=False)

    p2 = st.status("Phase 2: Training final model...", expanded=True)
    with p2:
        tb = st.progress(0); tt = st.empty()
        def tcb(e, t, l): tb.progress(min(e/t, 1.0)); tt.write(f"Epoch **{e}/{t}** -- Loss: {l:.6f}")
        model, ath = train_final(idim, best, norm_tr, fin_ep, tcb)
        tb.progress(1.0); tt.empty(); fth = ath * 3
        st.write(f"**Alert:** {ath:.6f}"); st.write(f"**Critical:** {fth:.6f}"); st.write(f"**Params:** {sum(p.numel() for p in model.parameters()):,}")
    p2.update(label="Phase 2: Model trained", state="complete", expanded=False)

    st.divider()
    c1, c2 = st.columns([2,1])
    with c1: st.caption("ONBOARD TIME / STEP"); td = st.empty()
    with c2: st.caption("AI RUL PREDICTION"); rd = st.empty()
    st.divider()
    cm, ci = st.columns([3,1])
    with cm:
        st.subheader("HEALTH TREND"); hp = st.empty()
        if ds.has_vibration: st.subheader("SPECTRAL SIGNATURE (FFT)"); st.caption("BPFO ~166 Hz, BPFI ~236 Hz"); fp = st.empty()
        elif ds.anomaly_labels is not None: st.subheader("ANOMALY vs GROUND TRUTH"); st.caption("Red = labeled anomalies"); gp = st.empty()
    with ci:
        st.subheader("SYSTEM INFO")
        st.markdown(f'<div class="metric-card"><b>Dataset:</b> {ds.name}<br><b>Format:</b> {ds.format_type}<br><b>Model:</b> PyTorch AE<br><b>Arch:</b> {best.layers}<br><b>Act:</b> {best.activation}<br><b>Device:</b> {DEVICE}</div>', unsafe_allow_html=True)
        st.subheader("DIAGNOSTIC LOG"); lp = st.empty()
        if ds.has_vibration: st.subheader("FFT STATUS"); fsp = st.empty()

    buf = deque(maxlen=win); hist = []; rul = RULPredictor(fth, 15); model.eval()
    norm_te = sc.transform(ds.test_data)

    for step in range(len(norm_te)):
        try:
            tl = ds.time_labels[step] if step < len(ds.time_labels) else f"Step {step}"
            td.markdown(f'<div class="time-display">{tl}</div>', unsafe_allow_html=True)
            x = torch.FloatTensor(norm_te[step:step+1]).to(DEVICE)
            with torch.no_grad(): err = float(torch.mean((x - model(x))**2).item())
            buf.append(err); sm = sum(buf)/len(buf); hist.append(sm)
            rt, rc = fmt_rul(rul.update(hist)); rd.markdown(f"<div class='rul-display' style='color:{rc};'>{rt}</div>", unsafe_allow_html=True)
            sc2, sl = classify(sm, ath)
            lp.markdown(f'<div class="metric-card {sc2}"><b>STATUS: {sl}</b><br>Score: {sm:.6f}<br>Alert: {ath:.6f}<br>Critical: {fth:.6f}<br>Step: {len(hist)}</div>', unsafe_allow_html=True)
            f = plot_health(hist, ath, fth, rul); hp.pyplot(f, use_container_width=True); plt.close(f)
            if ds.has_vibration and ds.raw_test_signals and step < len(ds.raw_test_signals) and len(hist) % 6 == 0:
                r = spectral_analysis(ds.raw_test_signals[step], ds.sample_rate)
                ff = plot_fft(r, len(ds.raw_test_signals[step]), ds.sample_rate); fp.pyplot(ff, use_container_width=True); plt.close(ff)
                if r["confirmed"]:
                    faults = []
                    if r["bpfo_det"]: faults.append(f"BPFO ({r['bpfo_amp']:.4f})")
                    if r["bpfi_det"]: faults.append(f"BPFI ({r['bpfi_amp']:.4f})")
                    fsp.markdown(f'<div class="metric-card status-crit"><b>FAULT CONFIRMED</b><br>Detected: {", ".join(faults)}<br>Validation: POSITIVE</div>', unsafe_allow_html=True)
                else:
                    fsp.markdown(f'<div class="metric-card status-ok"><b>NO FAULT</b><br>BPFO: {r["bpfo_amp"]:.4f}<br>BPFI: {r["bpfi_amp"]:.4f}<br>Dominant: {r["dom_freq"]:.1f} Hz</div>', unsafe_allow_html=True)
            elif ds.anomaly_labels is not None and len(hist) % 4 == 0:
                la = ds.anomaly_labels[:len(hist)] if len(ds.anomaly_labels) >= len(hist) else None
                fg = plot_gt(hist, la, ath, fth); gp.pyplot(fg, use_container_width=True); plt.close(fg)
            time.sleep(speed / 1000)
        except Exception as e: lp.error(f"Error step {step}: {e}"); break
    st.success("Telemetry replay complete.")