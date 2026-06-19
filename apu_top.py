#!/usr/bin/env python3
"""xdnatop — High-freq TUI for AMD GPU + XDNA NPU with per-engine bars."""

from __future__ import annotations

import argparse
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.align import Align
from rich.box import ROUNDED, DOUBLE
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.style import Style
from rich.table import Table
from rich.text import Text

GPU_SYS = Path("/sys/class/drm/card0/device")
HWMAN = next(GPU_SYS.glob("hwmon/hwmon*"), None)
GPU_METRICS = GPU_SYS / "gpu_metrics"
ACCEL_SYS = Path("/sys/class/accel/accel0/device")

C_RED = "#ed174a"
C_ORANGE = "#ff6b35"
C_GOLD = "#f7c948"
C_CYAN = "#00d4ff"
C_MAG = "#b44aeb"
C_GREEN = "#00e676"
C_DIM = "#888888"
C_WHITE = "#f0f0f0"
C_BG = "#0d1117"
C_BLUE = "#3b82f6"
C_TEAL = "#14b8a6"

S = lambda c: Style(color=c)
BS = lambda c: Style(bold=True, color=c)

BAR_COLORS = [C_RED, C_ORANGE, C_GOLD, C_CYAN, C_GREEN]


@dataclass
class GpuEng:
    gfx: float = 0.0
    media: float = 0.0
    vram_used_mb: float = 0.0
    vram_total_mb: float = 0.0
    temp_edge: float = 0.0
    temp_hotspot: float = 0.0
    temp_vram: float = 0.0
    power_w: float = 0.0
    sclk_mhz: float = 0.0
    name: str = ""
    cu_count: int = 0
    ipu_tiles: list[float] = field(default_factory=lambda: [0.0] * 8)
    ipu_power_mw: float = 0.0
    ipu_clk_mhz: float = 0.0
    ipu_reads_mbs: float = 0.0
    ipu_writes_mbs: float = 0.0
    dram_reads_mbs: float = 0.0
    dram_writes_mbs: float = 0.0
    soc_temp: float = 0.0
    gfx_clk_mhz: float = 0.0
    apu_power_w: float = 0.0





@dataclass
class SysInfo:
    cpu_temp: float = 0.0
    cpu_freq: float = 0.0
    load_1: float = 0.0
    load_5: float = 0.0
    load_15: float = 0.0
    mem_total_gb: float = 0.0
    mem_used_gb: float = 0.0
    gtt_used_mb: float = 0.0
    gtt_total_mb: float = 0.0
    fan_rpm: tuple = (0.0, 0.0, 0.0)


@dataclass
class AllData:
    gpu: GpuEng = field(default_factory=GpuEng)
    sys: SysInfo = field(default_factory=SysInfo)
    npu_fw: str = ""
    freq_hz: float = 0.0
    timestamp: float = 0.0


_GPU_CACHE: GpuEng = GpuEng()


# ─── gpu_metrics binary parser ───────────────────────────────────────────────


def _parse_gpu_metrics() -> dict:
    try:
        data = GPU_METRICS.read_bytes()
    except (FileNotFoundError, OSError):
        return {}
    if len(data) < 4:
        return {}
    hdr = struct.unpack_from("<HBb", data, 0)
    fmt_rev = hdr[1]

    r = {}
    if fmt_rev == 3:
        if len(data) >= 46 + 16:
            raw = struct.unpack_from("<8H", data, 46)
            r["ipu_tiles"] = [float(v) for v in raw]
        if len(data) >= 44 + 2:
            vcn_raw = struct.unpack_from("<H", data, 44)[0]
            r["vcn_act"] = vcn_raw / 100.0 if vcn_raw != 0xFFFF else 0.0
        if len(data) >= 42 + 2:
            r["gfx_act"] = float(struct.unpack_from("<H", data, 42)[0])
        if len(data) >= 4 + 2:
            v = struct.unpack_from("<H", data, 4)[0]
            r["temp_gfx"] = v / 100.0 if v != 0xFFFF else 0.0
        if len(data) >= 6 + 2:
            v = struct.unpack_from("<H", data, 6)[0]
            r["temp_soc"] = v / 100.0 if v != 0xFFFF else 0.0
        if len(data) >= 94 + 2:
            r["dram_r"] = float(struct.unpack_from("<H", data, 94)[0])
            r["dram_w"] = float(struct.unpack_from("<H", data, 96)[0])
        if len(data) >= 98 + 2:
            r["ipu_r"] = float(struct.unpack_from("<H", data, 98)[0])
            r["ipu_w"] = float(struct.unpack_from("<H", data, 100)[0])
        # u64 system_clock_counter at 104 (8-byte aligned, 2B pad @ 102)
        if len(data) >= 112 + 4:
            r["sock_pow"] = struct.unpack_from("<I", data, 112)[0] / 1000.0
        if len(data) >= 116 + 2:
            r["ipu_pow"] = float(struct.unpack_from("<H", data, 116)[0])  # CORRECTED
        if len(data) >= 120 + 4:
            v = struct.unpack_from("<I", data, 120)[0]
            r["apu_pow"] = v / 1000.0 if v != 0xFFFFFFFF else 0.0
        if len(data) >= 124 + 4:
            v = struct.unpack_from("<I", data, 124)[0]
            r["gfx_pow"] = v / 1000.0 if v != 0xFFFFFFFF else 0.0
        if len(data) >= 174 + 2:
            v = struct.unpack_from("<H", data, 174)[0]
            r["gfx_clk"] = float(v) if v != 0xFFFF else 0.0
        if len(data) >= 180 + 2:
            v = struct.unpack_from("<H", data, 180)[0]
            r["ipu_clk"] = float(v) if v != 0xFFFF else 0.0
        if len(data) >= 188 + 2:
            v = struct.unpack_from("<H", data, 188)[0]
            r["mpipu_freq"] = float(v) if v != 0xFFFF else 0.0
    return r


# ─── GPU readers ─────────────────────────────────────────────────────────────


def _read(p: Path) -> str:
    try:
        return p.read_text().strip()
    except (FileNotFoundError, OSError):
        return "0"


def _rdint(p: Path) -> int:
    try:
        return int(_read(p))
    except ValueError:
        return 0


def read_gpu() -> GpuEng:
    d = GpuEng()
    d.gfx = float(_read(GPU_SYS / "gpu_busy_percent"))
    d.media = float(_read(GPU_SYS / "vcn_busy_percent"))

    if HWMAN:
        d.power_w = _rdint(HWMAN / "power1_average") / 1_000_000
        d.sclk_mhz = _rdint(HWMAN / "freq1_input") / 1_000_000
        d.temp_edge = _rdint(HWMAN / "temp1_input") / 1000

    gm = _parse_gpu_metrics()
    if gm:
        d.ipu_tiles = gm.get("ipu_tiles", [0.0] * 8)
        d.ipu_power_mw = gm.get("ipu_pow", 0.0)
        d.ipu_clk_mhz = gm.get("ipu_clk", 0.0)
        d.ipu_reads_mbs = gm.get("ipu_r", 0.0)
        d.ipu_writes_mbs = gm.get("ipu_w", 0.0)
        d.dram_reads_mbs = gm.get("dram_r", 0.0)
        d.dram_writes_mbs = gm.get("dram_w", 0.0)
        d.soc_temp = gm.get("temp_soc", 0.0)
        d.apu_power_w = gm.get("apu_pow", 0.0)
        if d.power_w == 0.0:
            d.power_w = gm.get("sock_pow", 0.0)
        if d.sclk_mhz == 0.0:
            d.sclk_mhz = gm.get("gfx_clk", 0.0)
        d.gfx_clk_mhz = gm.get("gfx_clk", 0.0)

    try:
        if not _GPU_CACHE.name:
            import amdsmi
            amdsmi.amdsmi_init()
            h = amdsmi.amdsmi_get_processor_handles()[0]
            info = amdsmi.amdsmi_get_gpu_asic_info(h)
            n = info.get("market_name", "Radeon Graphics")
            _GPU_CACHE.name = n.removeprefix("AMD ").removeprefix("amd ")
            _GPU_CACHE.cu_count = int(info.get("num_compute_units", 0))
            try:
                vu = amdsmi.amdsmi_get_gpu_vram_usage(h)
                _GPU_CACHE.vram_total_mb = float(vu.get("vram_total", 512))
            except Exception:
                pass
        d.name = _GPU_CACHE.name
        d.cu_count = _GPU_CACHE.cu_count
        d.vram_total_mb = _GPU_CACHE.vram_total_mb
        import amdsmi
        handles = amdsmi.amdsmi_get_processor_handles()
        vu = amdsmi.amdsmi_get_gpu_vram_usage(handles[0])
        d.vram_used_mb = float(vu.get("vram_used", 0))
        try:
            d.temp_hotspot = float(amdsmi.amdsmi_get_temp_metric(
                handles[0], amdsmi.AmdSmiTemperatureType.HOTSPOT, amdsmi.AmdSmiTemperatureMetric.CURRENT))
        except Exception:
            pass
        try:
            d.temp_vram = float(amdsmi.amdsmi_get_temp_metric(
                handles[0], amdsmi.AmdSmiTemperatureType.VRAM, amdsmi.AmdSmiTemperatureMetric.CURRENT))
        except Exception:
            pass
    except Exception:
        d.name = _GPU_CACHE.name or "Radeon Graphics"
        d.cu_count = _GPU_CACHE.cu_count
        d.vram_total_mb = _GPU_CACHE.vram_total_mb
    return d


# ─── NPU reader (fdinfo) ─────────────────────────────────────────────────────


# ─── Sys readers ─────────────────────────────────────────────────────────────


def read_sys() -> SysInfo:
    s = SysInfo()
    for hwmon in Path("/sys/class/hwmon").iterdir():
        try:
            name = (hwmon / "name").read_text().strip()
            if name == "k10temp":
                t = _rdint(hwmon / "temp1_input")
                if t:
                    s.cpu_temp = t / 1000
                    break
        except OSError:
            continue
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "MHz" in line:
                    s.cpu_freq = float(line.split(":")[1].strip())
                    break
    except OSError:
        pass
    try:
        with open("/proc/loadavg") as f:
            parts = f.read().split()
            s.load_1, s.load_5, s.load_15 = float(parts[0]), float(parts[1]), float(parts[2])
    except OSError:
        pass
    meminfo = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                meminfo[k.strip()] = int(v.strip().split()[0])
        total_kb = meminfo.get("MemTotal", 0)
        avail_kb = meminfo.get("MemAvailable", 0)
        s.mem_total_gb = total_kb / 1_000_000
        s.mem_used_gb = (total_kb - avail_kb) / 1_000_000
    except OSError:
        pass
    s.gtt_total_mb = _rdint(GPU_SYS / "mem_info_gtt_total") / 1_000_000
    s.gtt_used_mb = _rdint(GPU_SYS / "mem_info_gtt_used") / 1_000_000
    for hwmon in Path("/sys/class/hwmon").iterdir():
        try:
            name = (hwmon / "name").read_text().strip()
        except OSError:
            continue
        if name == "su_axb35":
            f = lambda i: float(_rdint(hwmon / f"fan{i}_input")) if (hwmon / f"fan{i}_input").exists() else 0.0
            s.fan_rpm = (f(1), f(2), f(3))
            break
    return s


def collect() -> AllData:
    return AllData(
        gpu=read_gpu(),
        sys=read_sys(),
        npu_fw=_read(ACCEL_SYS / "fw_version"),
        timestamp=time.time(),
    )


# ─── TUI ─────────────────────────────────────────────────────────────────────


def _bar(pct: float, w: int = 20) -> Text:
    """Fancy gradient bar with smooth character progression."""
    filled = pct / 100 * w
    full_blocks = int(filled)
    partial = filled - full_blocks
    out = Text()
    for i in range(w):
        r = i / max(w - 1, 1)
        ci = min(int(r * (len(BAR_COLORS) - 1)), len(BAR_COLORS) - 1)
        if i < full_blocks:
            ch = "█"
        elif i == full_blocks and partial > 0:
            # Smooth partial block: ▏▎▍▌▋▊▉█
            pidx = min(int(partial * 8), 7)
            ch = ["▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"][pidx]
        else:
            ch = "░"
        out.append(ch, Style(color=BAR_COLORS[ci], dim=i >= full_blocks))
    return out


def _tsty(t: float) -> str:
    return "green" if t < 50 else "yellow" if t < 70 else "orange3" if t < 85 else "red1"


def _gpu_panel(d: GpuEng) -> Panel:
    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(width=8)
    tbl.add_column(width=30)

    tbl.add_row(
        Text("GFX", style=BS(C_RED)),
        Text.assemble(_bar(d.gfx), "  ", Text(f"{d.gfx:.0f}%", style=BS(C_WHITE))),
    )
    tbl.add_row(
        Text("Media", style=BS(C_GREEN)),
        Text.assemble(_bar(d.media), "  ", Text(f"{d.media:.0f}%", style=BS(C_WHITE))),
    )

    vpct = (d.vram_used_mb / max(d.vram_total_mb, 1)) * 100
    tbl.add_row(
        Text("VRAM", style=BS(C_MAG)),
        Text.assemble(
            _bar(vpct), "  ",
            Text(f"{d.vram_used_mb:.0f}/{d.vram_total_mb:.0f} MB", style=BS(C_WHITE)),
        ),
    )

    # Temps row
    parts = []
    if d.temp_edge:
        parts.append(Text(f"E{d.temp_edge:.0f}°C", style=_tsty(d.temp_edge)))
    if d.temp_hotspot:
        parts.append(Text(f"HS{d.temp_hotspot:.0f}°C", style=_tsty(d.temp_hotspot)))
    if d.soc_temp:
        parts.append(Text(f"S{d.soc_temp:.0f}°C", style=_tsty(d.soc_temp)))
    if d.temp_vram:
        parts.append(Text(f"V{d.temp_vram:.0f}°C", style=_tsty(d.temp_vram)))
    if parts:
        tbl.add_row(Text("Temps", style=BS(C_ORANGE)), Group(*[p + "  " for p in parts]))

    # Power + Clock row
    info = Text.assemble(
        Text(f"{d.power_w:.1f}W", style=C_GOLD),
        Text("  •  ", style=C_DIM),
        Text(f"{d.gfx_clk_mhz:.0f}MHz" if d.gfx_clk_mhz else f"{d.sclk_mhz:.0f}MHz", style=C_CYAN),
    )
    tbl.add_row(Text("Pwr", style=BS(C_GOLD)), info)

    return Panel(
        Align.left(tbl),
        title=Text.assemble(
            Text("⬤", style=C_RED), "  ",
            Text(d.name or "Radeon Graphics", style=BS("white")),
            Text(f"  ({d.cu_count} CU)", style=C_DIM),
        ),
        border_style="bright_red",
        box=ROUNDED,
        padding=(1, 2),
    )


def _npu_panel(d: AllData) -> Panel:
    gpu = d.gpu
    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(width=8)
    tbl.add_column(width=30)

    active_tiles = sum(1 for t in gpu.ipu_tiles if t > 0)
    avg_tile = sum(gpu.ipu_tiles) / max(len(gpu.ipu_tiles), 1)

    # Summary bar
    tbl.add_row(
        Text("IPU", style=BS(C_CYAN)),
        Text.assemble(
            _bar(avg_tile), "  ",
            Text(f"{avg_tile:.0f}%", style=BS(C_WHITE)),
            Text(f"  · {active_tiles}/8 tiles", style=S(C_DIM)),
        ),
    )

    # 8 tiles in 2 rows × 4 cols
    tile_labels = ["T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7"]
    tile_colors = [C_RED, C_ORANGE, C_GOLD, C_GREEN, C_CYAN, C_BLUE, C_MAG, C_TEAL]

    for i in range(0, len(gpu.ipu_tiles), 2):
        parts = []
        for j in range(2):
            idx = i + j
            if idx >= len(gpu.ipu_tiles):
                break
            v = gpu.ipu_tiles[idx]
            t = tile_labels[idx]
            c = tile_colors[idx]
            parts.append(Text.assemble(
                _bar(v), " ", Text(t, style=BS(c)), " ", Text(f"{v:.0f}%", style=C_WHITE),
            ))
        tbl.add_row(
            Text("", style=C_DIM),
            Text.assemble(*[p + "  " for p in parts]),
        )

    # IPU clocks + power
    info_parts = []
    if gpu.ipu_clk_mhz:
        info_parts.append(Text(f"{gpu.ipu_clk_mhz:.0f} MHz", style=C_GOLD))
    if gpu.ipu_power_mw:
        info_parts.append(Text(f"{gpu.ipu_power_mw:.0f} mW", style=C_ORANGE))
    if info_parts:
        tbl.add_row(Text("", style=C_DIM), Text.assemble(*[p + "  " for p in info_parts]))

    # Memory bandwidth
    bw_parts = []
    if gpu.ipu_reads_mbs:
        bw_parts.append(Text(f"R {gpu.ipu_reads_mbs:.0f}", style=C_GREEN))
    if gpu.ipu_writes_mbs:
        bw_parts.append(Text(f"W {gpu.ipu_writes_mbs:.0f}", style=C_MAG))
    if bw_parts:
        s = Text.assemble(*[p + " MB/s  " for p in bw_parts])
        tbl.add_row(Text("IPU BW", style=S(C_DIM)), s)
    if gpu.dram_reads_mbs or gpu.dram_writes_mbs:
        dram_parts = []
        if gpu.dram_reads_mbs:
            dram_parts.append(Text(f"R {gpu.dram_reads_mbs:.0f}", style=C_GREEN))
        if gpu.dram_writes_mbs:
            dram_parts.append(Text(f"W {gpu.dram_writes_mbs:.0f}", style=C_MAG))
        s = Text.assemble(*[p + " MB/s  " for p in dram_parts])
        tbl.add_row(Text("DRAM BW", style=S(C_DIM)), s)

    if d.npu_fw:
        tbl.add_row(Text("FW", style=S(C_DIM)), Text(f"v{d.npu_fw[:16]}", style=C_DIM))

    return Panel(
        Align.left(tbl),
        title=Text.assemble(
            Text("◆", style=C_CYAN), "  ",
            Text("XDNA NPU", style=BS("white")),
            Text("  · 8 AIE Tiles", style=C_DIM) if active_tiles > 0 else Text("", style=C_DIM),
        ),
        border_style="cyan",
        box=ROUNDED,
        padding=(1, 2),
    )


def _sys_panel(d: SysInfo) -> Panel:
    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(width=8)
    tbl.add_column(width=35)

    cpu_s = Text.assemble(
        Text(f"{d.cpu_temp:.0f}°C", style=_tsty(d.cpu_temp)),
        Text(f"  {d.cpu_freq:.0f}MHz" if d.cpu_freq else "", style=C_CYAN),
        Text(f"  load {d.load_1:.1f}/{d.load_5:.1f}/{d.load_15:.1f}" if d.load_1 else "", style=C_DIM),
    )
    tbl.add_row(Text("CPU", style=BS(C_ORANGE)), cpu_s)

    mpct = (d.mem_used_gb / max(d.mem_total_gb, 1)) * 100
    if d.mem_total_gb:
        tbl.add_row(
            Text("RAM", style=BS(C_GREEN)),
            Text.assemble(
                _bar(mpct), "  ",
                Text(f"{d.mem_used_gb:.0f}/{d.mem_total_gb:.0f}GB", style=BS(C_WHITE)),
            ),
        )

    gpct = (d.gtt_used_mb / max(d.gtt_total_mb, 1)) * 100
    if d.gtt_total_mb:
        tbl.add_row(
            Text("GTT", style=BS(C_BLUE)),
            Text.assemble(
                _bar(gpct), "  ",
                Text(f"{d.gtt_used_mb:.0f}/{d.gtt_total_mb:.0f}MB", style=BS(C_WHITE)),
            ),
        )

    if any(r > 0 for r in d.fan_rpm):
        fan_max = 4000
        fan_parts = []
        for i, r in enumerate(d.fan_rpm):
            if r > 0:
                pct = min(r / fan_max * 100, 100)
                label = ["F1 ", "F2 ", "F3 "][i]
                fan_parts.append(Text.assemble(
                    _bar(pct), " ",
                    Text(f"{label}{r:.0f}", style=C_CYAN),
                ))
        tbl.add_row(Text("Fan", style=BS(C_CYAN)), Text.assemble(*[p + "  " for p in fan_parts]))

    return Panel(
        Align.left(tbl),
        title=Text.assemble(Text("⚙", style=C_GOLD), "  ", Text("System", style=BS("white"))),
        border_style="gold3",
        box=ROUNDED,
        padding=(1, 2),
    )


def _layout() -> Layout:
    ly = Layout()
    ly.split(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=1),
    )
    ly["main"].split_row(
        Layout(name="left"),
        Layout(name="right"),
    )
    ly["main"]["left"].split(
        Layout(name="gpu"),
        Layout(name="sys"),
    )
    ly["main"]["right"].split(
        Layout(name="npu"),
    )
    return ly


def _header(d: AllData) -> Panel:
    freq_s = f" • {d.freq_hz:.0f} fps" if d.freq_hz else ""
    txt = Text.assemble(
        Text("  ⎔ xdnatop  ", style=BS(C_RED)),
        Text("▸  ", style=S(C_ORANGE)),
        Text(time.strftime("%H:%M:%S"), style=S(C_DIM)),
        Text(freq_s, style=S(C_DIM)),
    )
    return Panel(txt, box=DOUBLE, border_style="bright_red", style=S(C_BG))


def _footer() -> Panel:
    return Panel(
        Align.right(
            Text.assemble(
                Text(" q ", style=Style(bold=True, color="white", bgcolor="red")),
                Text(" quit  ", style=S(C_DIM)),
                Text(" f ", style=Style(bold=True, color="black", bgcolor=C_CYAN)),
                Text(" high-freq  ", style=S(C_DIM)),
            ),
        ),
        box=ROUNDED,
        style=S(C_BG),
        border_style=S(C_BG),
    )


def render(d: AllData) -> Layout:
    ly = _layout()
    ly["header"].update(_header(d))
    ly["gpu"].update(_gpu_panel(d.gpu))
    ly["npu"].update(_npu_panel(d))
    ly["sys"].update(_sys_panel(d.sys))
    ly["footer"].update(_footer())
    return ly


# ─── Watch mode ──────────────────────────────────────────────────────────────


def _watch(high_freq: bool = False):
    import signal
    import sys

    signal.signal(signal.SIGINT, lambda s, f: exit(0))
    interval = 0.1 if high_freq else 0.5
    while True:
        d = collect()
        gpu_s = f"GFX:{d.gpu.gfx:.0f}% VRAM:{d.gpu.vram_used_mb:.0f}MB"
        ipu_avg = sum(d.gpu.ipu_tiles) / max(len(d.gpu.ipu_tiles), 1)
        ipu_s = f"IPU:{ipu_avg:.0f}%"
        fan_s = f" fan:{d.sys.fan_rpm[0]:.0f}" if d.sys.fan_rpm[0] else ""
        line = (f"\r{gpu_s:<32} {ipu_s:<14} |"
                f" CPU:{d.sys.cpu_temp:.0f}°C load:{d.sys.load_1:.1f}{fan_s} | {time.strftime('%H:%M:%S')}")
        sys.stdout.write(line)
        sys.stdout.flush()
        time.sleep(interval)


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="High-freq TUI for AMD GPU + XDNA NPU")
    parser.add_argument("-w", "--watch", action="store_true", help="Simple watch mode (no TUI)")
    parser.add_argument("-f", "--freq", action="store_true", help="High-frequency mode (20 fps)")
    parser.add_argument("-F", "--ultra", action="store_true", help="Ultra high-frequency mode (60 fps)")
    args = parser.parse_args()

    import signal
    signal.signal(signal.SIGINT, lambda s, f: exit(0))

    if args.watch:
        _watch(high_freq=args.freq or args.ultra)
        return

    fps = 60 if args.ultra else (20 if args.freq else 4)
    try:
        with Live(render(collect()), screen=True, refresh_per_second=fps) as live:
            while True:
                d = collect()
                d.freq_hz = fps
                live.update(render(d))
                time.sleep(1.0 / fps)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
