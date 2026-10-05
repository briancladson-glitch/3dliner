"""PDF survey report (reportlab + matplotlib).

Layout: 1 summary · 2 measuring-position table + diameter profile · 3 wall map + cross-sections ·
4 3D view + quality & traceability · 5 method, notes, sign-off · appendix: every 30 mm band.
"""

from __future__ import annotations

import datetime as _dt
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from .calibration import DeviceCalibration  # noqa: E402
from .config import EngineSpec, JobInfo, ReportLimits  # noqa: E402
from .pipeline import BandResult, ScanResult  # noqa: E402

# ----------------------------------------------------------------------------- style
NAVY = "#14304f"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
RULE = "#d9d8d3"
PANEL = "#f4f3ef"
FA_C = "#2a78d6"  # categorical slot 1 (blue)
PS_C = "#eb6834"  # categorical slot 2 (orange)
ENV_C = "#c9c8c2"
LIMIT_C = "#c62828"  # status: critical
OK_C = "#2e7d32"  # status: good
WARN_C = "#b26a00"  # status: warning
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _fonts() -> tuple[str, str]:
    d = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    try:
        pdfmetrics.registerFont(TTFont("DejaVu", str(d / "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(d / "DejaVuSans-Bold.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVu-Oblique", str(d / "DejaVuSans-Oblique.ttf")))
        from reportlab.pdfbase.pdfmetrics import registerFontFamily
        registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu-Oblique",
                           boldItalic="DejaVu-Bold")
        return "DejaVu", "DejaVu-Bold"
    except Exception:  # pragma: no cover - fallback if the TTF is missing
        return "Helvetica", "Helvetica-Bold"


FONT, FONT_B = _fonts()

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": RULE, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 8.5,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "grid.color": "#ecebe7", "grid.linewidth": 0.6,
    "legend.frameon": False, "legend.fontsize": 7, "figure.dpi": 220, "savefig.dpi": 220,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _ps(name, size=8.5, font=None, color=INK, lead=None, align=TA_LEFT, **kw):
    return ParagraphStyle(name, fontName=font or FONT, fontSize=size, textColor=colors.HexColor(color),
                          leading=lead or size * 1.32, alignment=align, **kw)


S = {
    "h1": _ps("h1", 17, FONT_B, NAVY, 21),
    "h2": _ps("h2", 11, FONT_B, NAVY, 14, spaceBefore=4, spaceAfter=4),
    "h3": _ps("h3", 8.8, FONT_B, INK, 11, spaceBefore=3, spaceAfter=2),
    "body": _ps("body", 8.3),
    "small": _ps("small", 7.2, color=INK2),
    "tiny": _ps("tiny", 6.4, color=INK2),
    "cell": _ps("cell", 7.2, lead=8.6),
    "cellb": _ps("cellb", 7.2, FONT_B, lead=8.6),
    "cellr": _ps("cellr", 7.2, lead=8.6, align=TA_RIGHT),
    "kpi_v": _ps("kpi_v", 15, FONT_B, NAVY, 17),
    "kpi_l": _ps("kpi_l", 6.8, FONT_B, INK2, 8.5),
    "kpi_s": _ps("kpi_s", 6.6, color=INK2, lead=8.2),
    "center": _ps("center", 7.5, align=TA_CENTER, color=INK2),
}


def _f(x, n=2):
    return "—" if x is None or not np.isfinite(x) else f"{x:.{n}f}"


# ----------------------------------------------------------------------------- figures
def _depth_axis(ax, res: ScanResult, engine: EngineSpec):
    d0, d1 = res.depth_range
    ax.set_ylim(d1 + 20, max(d0 - 20, -10))
    ax.set_ylabel("Depth below liner top (mm)")
    if d1 > engine.port_top_depth:
        ax.axhspan(engine.port_top_depth, min(engine.port_bottom_depth, d1 + 20), color="#f4f3ef", zorder=0)
    ax.grid(True, axis="both")


def fig_profile(res: ScanResult, engine: EngineSpec, limits: ReportLimits, path: Path) -> Path:
    b = [x for x in res.profile]
    d = np.array([x.depth for x in b])
    fa = np.array([x.stats.D_FA for x in b])
    ps = np.array([x.stats.D_PS for x in b])
    dmax = np.array([x.stats.D_max for x in b])
    dmin = np.array([x.stats.D_min for x in b])
    ov = np.array([x.stats.ovality for x in b])
    mm_ = dmax - dmin
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 4.6), gridspec_kw={"width_ratios": [1.6, 1]}, sharey=True)
    a1.fill_betweenx(d, dmin, dmax, color=ENV_C, alpha=0.55, lw=0, label="min–max envelope")
    a1.plot(fa, d, color=FA_C, lw=1.6, label="D FA (fore–aft)")
    a1.plot(ps, d, color=PS_C, lw=1.6, label="D PS (port–stbd)")
    a1.axvline(engine.bore, color=MUTED, lw=1, ls=(0, (4, 3)))
    _depth_axis(a1, res, engine)
    top = a1.get_ylim()[1]
    a1.text(engine.bore, top, f" nominal {engine.bore:.0f}", color=INK2, fontsize=6.5, va="top")
    lim = engine.bore + limits.wear_limit(engine.bore)
    x0, x1 = engine.bore - 0.3, max(dmax.max(), engine.bore) + 0.35
    if lim < x1 + 0.5:
        a1.axvline(lim, color=LIMIT_C, lw=1, ls=(0, (2, 2)))
        a1.text(lim, top, " wear limit*", color=LIMIT_C, fontsize=6.5, va="top")
        x1 = max(x1, lim + 0.3)
    else:
        a1.text(0.99, 0.01, f"wear limit* {lim:.2f} mm: off scale", transform=a1.transAxes, ha="right",
                va="bottom", color=LIMIT_C, fontsize=6.3)
    a1.set_xlim(min(x0, dmin.min() - 0.1), x1)
    a1.set_xlabel("Diameter (mm, as measured)")
    a1.set_title("Diameter profile", loc="left")
    a1.legend(loc="lower right")
    from matplotlib.transforms import blended_transform_factory

    tr = blended_transform_factory(a1.transAxes, a1.transData)
    for p in res.positions:
        a1.plot([0.0], [p.depth], marker=">", color=NAVY, ms=3.5, clip_on=False, transform=tr)
    a2.plot(ov, d, color=NAVY, lw=1.6, label="ovality |PS − FA|")
    a2.plot(mm_, d, color=MUTED, lw=1.2, ls=(0, (3, 2)), label="D max − D min")
    a2.axvline(limits.ovality_limit_mm, color=LIMIT_C, lw=1, ls=(0, (2, 2)))
    a2.text(limits.ovality_limit_mm, res.depth_range[1] + 15, " limit*", color=LIMIT_C, fontsize=6.5, va="bottom")
    a2.set_xlim(0, max(limits.ovality_limit_mm * 1.15, mm_.max() * 1.15))
    a2.set_xlabel("mm")
    a2.set_title("Ovality", loc="left")
    a2.legend(loc="lower right")
    a2.grid(True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _seq_cmap():
    return LinearSegmentedColormap.from_list("seq_blue", SEQ)


def _fill_holes(dev: np.ndarray, max_run: int = 3) -> np.ndarray:
    """Display only: interpolate short empty runs (≤ max_run bins) along depth; long gaps (ports) stay empty."""
    d = dev.copy()
    idx = np.arange(d.shape[0])
    for j in range(d.shape[1]):
        col = d[:, j]
        bad = np.isnan(col)
        if not bad.any() or bad.all():
            continue
        edges = np.diff(np.concatenate([[0], bad.astype(int), [0]]))
        for a, b in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0], strict=False):
            if b - a <= max_run and a > 0 and b < col.size:
                col[a:b] = np.interp(idx[a:b], [a - 1, b], [col[a - 1], col[b]])
    return d


def _short_flags(b: BandResult) -> str:
    out = []
    for f in b.flags:
        out.append("ports" if f.startswith("scavenge") else "steep" if f.startswith("steep") else
                   "non-elliptical" if f.startswith("shape") else "gap" if f.startswith("angular") else f[:20])
    return ", ".join(out)


def fig_map(res: ScanResult, engine: EngineSpec, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    dev = _fill_holes(res.map_dev)
    vals = dev[np.isfinite(dev)]
    vmin, vmax = np.quantile(vals, [0.01, 0.995]) if vals.size else (0, 1)
    da = res.map_alpha[1] - res.map_alpha[0]
    dd = res.map_depth[1] - res.map_depth[0]
    ext = [res.map_alpha[0] - da / 2, res.map_alpha[-1] + da / 2, res.map_depth[-1] + dd / 2, res.map_depth[0] - dd / 2]
    cm = _seq_cmap()
    cm = cm.with_extremes(bad="#ffffff")
    im = ax.imshow(np.ma.masked_invalid(dev), aspect="auto", extent=ext, cmap=cm, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    if res.outlier_alpha.size:
        ax.scatter(res.outlier_alpha, res.outlier_depth, s=0.4, color=LIMIT_C, alpha=0.5, lw=0,
                   label="marked outliers (excluded)")
        ax.legend(loc="lower right", markerscale=8)
    ax.set_xticks([0, 90, 180, 270, 360], ["FORE\n0°", "PORT\n90°", "AFT\n180°", "STBD\n270°", "FORE\n360°"])
    ax.set_ylabel("Depth below liner top (mm)")
    ax.set_title("Unrolled wall map — radius minus nominal radius (as measured)", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.015)
    cb.set_label("mm", color=INK2)
    cb.outline.set_visible(False)
    if res.depth_range[1] > engine.port_top_depth:
        ax.axhline(engine.port_top_depth, color=INK2, lw=0.6, ls=(0, (3, 3)))
        ax.text(2, engine.port_top_depth - 8, "scavenge ports below", fontsize=6.3, color=INK2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _pick_sections(res: ScanResult, n: int = 4) -> list[BandResult]:
    cands = [b for b in res.positions if not b.port_zone] or res.positions
    if len(cands) <= n:
        return cands
    worst = max(cands, key=lambda b: b.stats.D_max)
    rest = [b for b in cands if b is not worst]
    idx = np.linspace(0, len(rest) - 1, n - 1).round().astype(int)
    out = sorted({id(worst): worst, **{id(rest[i]): rest[i] for i in idx}}.values(), key=lambda b: b.depth)
    return out[:n]


def fig_sections(res: ScanResult, engine: EngineSpec, path: Path) -> Path:
    from .fitting import FourierFit

    secs = _pick_sections(res)
    a = np.linspace(0, 2 * np.pi, 721)
    shapes = []
    for b in secs:
        f = FourierFit(8, b.coef, np.zeros((b.coef.size, b.coef.size)), b.sigma_r, b.n_points, b.coef.size > 17)
        shapes.append(f.rho(a) - b.coef[0])  # shape about the mean circle (odd harmonics = centre offset removed)
    for sh, b in zip(shapes, secs, strict=False):
        sh -= b.coef[1] * np.cos(a) + b.coef[2] * np.sin(a)
    full = max(0.02, max(np.abs(sh).max() for sh in shapes))
    full = float(np.ceil(full * 100) / 100)
    fig, axs = plt.subplots(1, len(secs), figsize=(7.2, 2.4), subplot_kw={"projection": "polar"})
    axs = np.atleast_1d(axs)
    for ax, b, sh in zip(axs, secs, shapes, strict=False):
        r = 1.0 + 0.45 * sh / full
        ax.fill_between(a, 0, r, color=SEQ[0], lw=0)
        ax.plot(a, np.ones_like(a), color=MUTED, lw=0.8, ls=(0, (3, 2)))
        ax.plot(a, r, color=NAVY, lw=1.3)
        ax.set_ylim(0, 1.5)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(1)
        ax.set_xticks(np.radians([0, 90, 180, 270]), ["F", "P", "A", "S"])
        ax.set_yticks([0.55, 1.0, 1.45])
        ax.set_yticklabels([])
        ax.grid(color="#ecebe7")
        ax.set_title(f"{b.depth:.0f} mm\nD̄ {b.stats.D_mean:.2f} · ov {b.stats.ovality:.2f}", fontsize=7, pad=6)
    fig.text(0.5, 0.01, f"Shape about the mean circle (dashed). Grid rings at ±{full:.2f} mm radial — "
             "deviations magnified for display.", ha="center", fontsize=6.8, color=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_3d(res: ScanResult, engine: EngineSpec, path: Path) -> Path:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    dev = res.map_dev.copy()
    # fill small holes along angle for a continuous surface
    for i in range(dev.shape[0]):
        row = dev[i]
        m = np.isfinite(row)
        if m.sum() > 10:
            dev[i] = np.interp(np.arange(row.size), np.nonzero(m)[0], row[m], period=row.size)
    A, D = np.meshgrid(np.radians(res.map_alpha), res.map_depth)
    R = engine.bore / 2
    X, Y, Z = R * np.cos(A), R * np.sin(A), -D
    vals = dev[np.isfinite(dev)]
    vmin, vmax = np.quantile(vals, [0.01, 0.995])
    cm = _seq_cmap()
    fc = cm(np.clip((np.nan_to_num(dev, nan=vmin) - vmin) / (vmax - vmin), 0, 1))
    fig = plt.figure(figsize=(3.4, 4.3))
    ax = fig.add_subplot(111, projection="3d")
    sel = A[0] <= np.radians(270)  # cut-away view
    ax.plot_surface(X[:, sel], Y[:, sel], Z[:, sel], facecolors=fc[:, sel], rstride=2, cstride=2, linewidth=0,
                    antialiased=False, shade=False)
    ax.set_box_aspect((1, 1, (res.depth_range[1] - res.depth_range[0]) / engine.bore))
    ax.view_init(elev=24, azim=-58)
    ax.set_axis_off()
    ax.set_title("3D scan (cut-away), colour as wall map", fontsize=7.5, color=INK2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------- page furniture
@dataclass
class _Ctx:
    report_no: str
    device: str
    cal_id: str
    simulated: bool
    generated: str


def _on_page(ctx: _Ctx):
    def draw(c, doc):
        W, H = A4
        c.saveState()
        if ctx.simulated:
            c.setFont(FONT_B, 54)
            c.setFillColor(colors.Color(0.78, 0.16, 0.16, alpha=0.07))
            c.translate(W / 2, H / 2)
            c.rotate(35)
            c.drawCentredString(0, 0, "DRAFT · SIMULATED")
            c.rotate(-35)
            c.translate(-W / 2, -H / 2)
        c.setFillColor(colors.HexColor(NAVY))
        c.rect(0, H - 13 * mm, W, 13 * mm, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.setFont(FONT_B, 10.5)
        c.drawString(16 * mm, H - 8.3 * mm, "UMA MARINE")
        c.setFont(FONT, 8)
        c.drawString(46 * mm, H - 8.3 * mm, "LinerScan · Cylinder liner bore survey")
        c.drawRightString(W - 16 * mm, H - 8.3 * mm, f"Report {ctx.report_no}   ·   page {doc.page}")
        c.setStrokeColor(colors.HexColor(RULE))
        c.line(16 * mm, 12 * mm, W - 16 * mm, 12 * mm)
        c.setFont(FONT, 6.5)
        c.setFillColor(colors.HexColor(INK2))
        c.drawString(16 * mm, 8 * mm, f"Instrument {ctx.device} · calibration {ctx.cal_id} · generated {ctx.generated}")
        if ctx.simulated:
            c.setFillColor(colors.HexColor(LIMIT_C))
            c.setFont(FONT_B, 6.5)
            c.drawRightString(W - 16 * mm, 8 * mm, "DRAFT — SIMULATED DATA — NOT A SURVEY RECORD")
        c.restoreState()

    return draw


def _table(data, widths, header=True, zebra=True, align_right_from=1, font_size=7.2, extra=None):
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [
        ("FONT", (0, 0), (-1, -1), FONT, font_size),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor(INK)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, colors.HexColor(RULE)),
    ]
    if align_right_from is not None:
        st.append(("ALIGN", (align_right_from, 1 if header else 0), (-1, -1), "RIGHT"))
    if header:
        st += [("FONT", (0, 0), (-1, 0), FONT_B, font_size - 0.3),
               ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
               ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
               ("ALIGN", (0, 0), (-1, 0), "CENTER")]
    if zebra:
        for i in range(1 if header else 0, len(data)):
            if i % 2 == 0:
                st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor(PANEL)))
    if extra:
        st += extra
    t.setStyle(TableStyle(st))
    return t


def _kv_table(rows, widths):
    data = [[Paragraph(k, S["small"]), Paragraph(v, S["cell"])] for k, v in rows]
    t = Table(data, colWidths=widths)
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 1.4),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 1.4), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#ecebe7"))]))
    return t


def _panel(flow, width, title=None):
    content = ([Paragraph(title, S["h3"])] if title else []) + flow
    t = Table([[content]], colWidths=[width])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(PANEL)),
                           ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                           ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return t


def _kpi(label, value, sub, accent=NAVY):
    return [Paragraph(label.upper(), S["kpi_l"]), Paragraph(f'<font color="{accent}">{value}</font>', S["kpi_v"]),
            Paragraph(sub, S["kpi_s"])]


def _status(frac: float) -> tuple[str, str]:
    if frac >= 1.0:
        return "EXCEEDS LIMIT", LIMIT_C
    if frac >= 0.8:
        return "APPROACHING LIMIT", WARN_C
    return "WITHIN LIMIT", OK_C


# ----------------------------------------------------------------------------- the document
def build_report(res: ScanResult, cal: DeviceCalibration, engine: EngineSpec, job: JobInfo, limits: ReportLimits,
                 out_path: str | Path, plan=None, simulated: bool = True, workdir: str | Path | None = None,
                 extra_notes: list[str] | None = None) -> Path:
    out_path = Path(out_path)
    wd = Path(workdir or out_path.parent / (out_path.stem + "_fig"))
    wd.mkdir(parents=True, exist_ok=True)
    figs = {"profile": fig_profile(res, engine, limits, wd / "profile.png"),
            "map": fig_map(res, engine, wd / "map.png"),
            "sections": fig_sections(res, engine, wd / "sections.png"),
            "3d": fig_3d(res, engine, wd / "3d.png")}
    raw_meta = res.meta.get("raw", {})
    ctx = _Ctx(job.report_no, raw_meta.get("device", "LS-?"), cal.cal_id, simulated,
               _dt.datetime.now().strftime("%Y-%m-%d %H:%M"))
    W, H = A4
    doc = BaseDocTemplate(str(out_path), pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=19 * mm,
                          bottomMargin=16 * mm, title=f"Liner bore survey {job.report_no}", author="UMA Marine",
                          subject="LinerScan cylinder liner survey")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
    doc.addPageTemplates([PageTemplate("p", [frame], onPage=_on_page(ctx))])
    CW = doc.width
    story = []

    main = [b for b in res.profile if not b.port_zone]
    allb = main or res.profile
    worst_D = max(allb, key=lambda b: b.stats.D_max)
    worst_ov = max(allb, key=lambda b: b.stats.ovality)
    worst_mm = max(allb, key=lambda b: b.stats.max_minus_min)
    wear = worst_D.stats.D_max - engine.bore
    wlim = limits.wear_limit(engine.bore)
    T_l = job.liner_temperature_c
    therm = engine.bore * engine.liner_cte * (T_l - 20.0) if T_l is not None else float("nan")
    U = max((b.U_k2 for b in allb), default=float("nan"))

    # ------------------------------------------------------------------ page 1: summary
    story += [Paragraph("Cylinder liner bore survey", S["h1"]),
              Paragraph(f"Unit {job.cylinder} · {job.vessel} · {job.engine_maker_type}", _ps("sub", 9.5, color=INK2)),
              Spacer(1, 4 * mm)]
    left = _kv_table([
        ("Vessel / IMO", f"{job.vessel} / {job.imo}"), ("Engine", job.engine_maker_type),
        ("Engine serial", job.engine_serial), ("Cylinder unit", job.cylinder),
        ("Running hours / since overhaul", f"{job.running_hours} / {job.hours_since_overhaul}"),
        ("Client", job.client), ("Location / date", f"{job.location} / {job.date}"),
        ("Surveyor", job.surveyor)], [36 * mm, CW / 2 - 40 * mm])
    ev = {e[2].split()[0]: e[0] for e in res.meta.get("events", []) if e[1] == "state"}
    dur = ev.get("STOPPED", 0) - ev.get("SCAN", 0)
    right = _kv_table([
        ("Instrument", f"LinerScan {raw_meta.get('device', '')}, fw {raw_meta.get('firmware', '')}"),
        ("Optical module", f"preset {cal.preset} · calibration {cal.cal_id} ({cal.date})"),
        ("Nominal bore", f"Ø {engine.bore:.2f} mm"),
        ("Liner temperature", f"{_f(T_l, 0)} °C (jacket water) · instrument {res.temps['camera_scan']:.1f} °C"),
        ("Scan", f"1 stroke BDC → {job.stop_angle_read_deg}° BTDC in {dur:.0f} s · {res.n_points / 1e6:.2f} M points"),
        ("Scanned depth", f"{res.depth_range[0]:.0f} – {res.depth_range[1]:.0f} mm below liner top"),
        ("Device placement", f"offset {np.hypot(*allb[0].centre_xy):.1f} mm, tilt {res.tilt_deg[0]:.2f}° / "
                             f"{res.tilt_deg[1]:.2f}° (removed by fit)"),
        ("Orientation", f"FORE at {res.yaw_deg:.1f}° device angle ({res.yaw_source.split('(')[0].strip()})")],
        [30 * mm, CW / 2 - 34 * mm])
    t = Table([[left, right]], colWidths=[CW / 2, CW / 2])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    story += [t, Spacer(1, 5 * mm), Paragraph("Key results", S["h2"])]
    k = [
        _kpi("Largest diameter", f"{worst_D.stats.D_max:.2f}",
             f"mm at {worst_D.depth:.0f} mm depth, {worst_D.stats.D_max_angle_deg:.0f}°/"
             f"{(worst_D.stats.D_max_angle_deg + 180) % 360:.0f}°"),
        _kpi("Increase vs nominal", f"+{wear:.2f}",
             f"mm (as measured) = {100 * wear / wlim:.0f}% of limit*" if wlim else "mm"),
        _kpi("Max ovality |PS−FA|", f"{worst_ov.stats.ovality:.2f}",
             f"mm at {worst_ov.depth:.0f} mm · "
             f"{100 * worst_ov.stats.ovality / limits.ovality_limit_mm:.0f}% of limit*"),
        _kpi("Max D max − D min", f"{worst_mm.stats.max_minus_min:.2f}", f"mm at {worst_mm.depth:.0f} mm depth"),
    ]
    kt = Table([k], colWidths=[CW / 4] * 4)
    kt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(PANEL)),
                            ("LINEBEFORE", (1, 0), (-1, -1), 2, colors.white), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                            ("LEFTPADDING", (0, 0), (-1, -1), 8)]))
    story += [kt, Spacer(1, 5 * mm)]

    # condition assessment (DRAFT wording – to be confirmed by UMA)
    fw, cw = wear / wlim, worst_ov.stats.ovality / limits.ovality_limit_mm
    sw, cwc = _status(fw)
    so, coc = _status(cw)
    cond = [["Criterion", "Measured (worst band)", "Limit*", "Used", "Assessment"],
            [Paragraph("Diameter increase (D max − nominal)", S["cell"]), f"{wear:.2f} mm @ {worst_D.depth:.0f} mm",
             f"{wlim:.2f} mm", f"{100 * fw:.0f}%", Paragraph(f'<font color="{cwc}"><b>{sw}</b></font>', S["cell"])],
            [Paragraph("Ovality |D PS − D FA|", S["cell"]),
             f"{worst_ov.stats.ovality:.2f} mm @ {worst_ov.depth:.0f} mm",
             f"{limits.ovality_limit_mm:.2f} mm", f"{100 * cw:.0f}%",
             Paragraph(f'<font color="{coc}"><b>{so}</b></font>', S["cell"])]]
    story += [Paragraph("Condition against limits (draft)", S["h2"]),
              _table(cond, [62 * mm, 38 * mm, 22 * mm, 14 * mm, CW - 136 * mm], align_right_from=None),
              Spacer(1, 2.5 * mm)]
    notes = [
        f"* Limits are <b>placeholders</b> ({limits.wear_limit_pct:.1f}% of bore for diameter increase, "
        f"{limits.ovality_limit_mm:.2f} mm ovality) and must be replaced with the engine maker's values before "
        "this report is issued. The wording and the 80% \"approaching\" threshold are a draft for UMA's decision.",
        f"Diameters are stated <b>as measured</b> at a liner temperature of {_f(T_l, 0)} °C; no correction to 20 °C "
        f"is applied. For reference, thermal expansion of a cast-iron liner from 20 °C to {_f(T_l, 0)} °C is about "
        f"+{therm:.2f} mm on the diameter and is contained in the increase shown above.",
        f"Expanded uncertainty of a band diameter U(k=2) ≤ {U:.3f} mm (statistical + systematic budget, see page 4).",
    ]
    story += [Paragraph(n, S["small"]) for n in notes]
    se0 = raw_meta.get("safety_events", [])
    n_flag = sum(1 for b in res.profile if b.flags)
    glance = [
        Paragraph("Scan at a glance", S["h3"]),
        Paragraph(f"One continuous stroke recorded {res.n_points / 1e6:.2f} million wall points from "
                  f"{res.depth_range[0]:.0f} to {res.depth_range[1]:.0f} mm below the liner top: "
                  f"{len(res.profile)} bands of 30 mm, each measured all around the bore.", S["small"]),
        Spacer(1, 2),
        Paragraph(f"• Coverage: every non-port band ≥ {min((b.coverage_pct for b in allb), default=0):.0f}% of "
                  f"1° sectors.<br/>• Outliers marked: {100 * res.n_outliers / max(res.n_points, 1):.2f}% of points."
                  "<br/>• Ring check: "
                  + ("; ".join(f"{r['when']} {r['error'] * 1000:+.0f} µm" for r in cal.ring_checks) or "not recorded") +
                  f"<br/>• Flagged bands: {n_flag} (ports, ridge, shape) — see appendix."
                  f"<br/>• Crush safety: " + ("no warning or alarm." if not se0 else
                                              ", ".join(sorted({s_[1] for s_ in se0})) + " (see page 4).")
                  + "<br/>• Thermal compensation active, instrument at "
                  f"{res.temps['camera_scan']:.1f} °C.", S["small"]),
    ]
    g = Table([[Image(str(figs["3d"]), width=CW * 0.30, height=CW * 0.30 * 4.3 / 3.4), glance]],
              colWidths=[CW * 0.33, CW * 0.67])
    g.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    story += [Spacer(1, 4 * mm), g]

    # ------------------------------------------------------------------ page 2: table + profile
    story += [PageBreak(), Paragraph("Diameters at the standard measuring positions", S["h2"]),
              Paragraph("Each value is a 30 mm band (±15 mm about the stated depth) fitted with 3D data all around "
                        "the bore; diameters are taken across the fitted liner centre. Angles from FORE, "
                        "counter-clockwise seen from above (PORT 90°).", S["small"]), Spacer(1, 2 * mm)]
    hdr = ["Pos", "Depth\nmm", "D FA\nmm", "D PS\nmm", "Ovality\n|PS−FA|", "D max\nmm", "at\n°", "D min\nmm",
           "at\n°", "max−min\nmm", "Δ nominal\nmm", "U(k=2)\nmm", "Note"]
    rows = [hdr]
    extra = []
    for i, b in enumerate(res.positions, 1):
        st = b.stats
        note = _short_flags(b).replace("non-elliptical", "shape")
        rows.append([str(i), f"{b.depth:.0f}", f"{st.D_FA:.2f}", f"{st.D_PS:.2f}", f"{st.ovality:.2f}",
                     f"{st.D_max:.2f}", f"{st.D_max_angle_deg:.0f}", f"{st.D_min:.2f}", f"{st.D_min_angle_deg:.0f}",
                     f"{st.max_minus_min:.2f}", f"{st.D_mean - engine.bore:+.2f}", f"{b.U_k2:.3f}", note])
        if b is worst_D or (b.stats.D_max >= worst_D.stats.D_max - 1e-9):
            extra.append(("FONT", (5, i), (5, i), FONT_B, 7.2))
    cwid = [8, 13, 15.5, 15.5, 14, 15.5, 9, 15.5, 9, 14, 15, 13]
    cwid = [c * mm for c in cwid]
    cwid.append(CW - sum(cwid))
    story += [_table(rows, cwid, extra=extra), Spacer(1, 1.5 * mm),
              Paragraph("Notes: ports = band in the scavenge-port zone (diameters across the port bars only); "
                        "steep = diameter changes > 0.1 mm across the band (ring-reversal ridge), the value depends "
                        "on the exact height; shape = cross-section not elliptical (ellipse cross-check differs "
                        "from the Fourier result by > 0.05 mm). Δ nominal = mean diameter − nominal.", S["tiny"]),
              Spacer(1, 3 * mm), Image(str(figs["profile"]), width=CW, height=CW * 4.6 / 7.2),
              Paragraph("Arrows mark the standard positions; shaded = scavenge-port zone. Red dashed lines are "
                        "placeholder limits (*).", S["tiny"])]

    # ------------------------------------------------------------------ page 3: map + sections
    story += [PageBreak(), Paragraph("Wall map and cross-sections", S["h2"]),
              Image(str(figs["map"]), width=CW, height=CW * 4.4 / 7.2),
              Paragraph("Every scan point within each 30 mm band, binned 2° × 10 mm. Red dots are points marked as "
                        f"outliers (beyond max(5·1.4826·MAD, 0.1 mm) of the band fit): {res.n_outliers:,} of "
                        f"{res.n_points:,} ({100 * res.n_outliers / max(res.n_points, 1):.2f}%). They are excluded "
                        "from the fits but kept in the data set; clusters of them mark real local features such "
                        "as scuffing.", S["tiny"]), Spacer(1, 3 * mm),
              Image(str(figs["sections"]), width=CW, height=CW * 2.3 / 7.2)]

    # ------------------------------------------------------------------ page 4: 3D + QA
    tri = cal.triangulation
    rc = cal.ring_checks
    ring_txt = "; ".join(f"{r['when']}: {r['error'] * 1000:+.1f} µm at {r['T']:.0f} °C" for r in rc) or "not recorded"
    flagged = [b for b in res.profile if b.flags]
    gaps = max((b.max_gap_deg for b in allb), default=0)
    cov = min((b.coverage_pct for b in allb), default=0)
    se = raw_meta.get("safety_events", [])
    radio = raw_meta.get("radio", {})
    qa = _kv_table([
        ("Triangulation model", f"{tri.kind}, {tri.structure.replace('_', '-')} (leave-one-out RMS "
                                f"{1000 * min(tri.loo_rms.values()):.1f} µm; chosen by the ≥20 % and ≥1 µm rule)"),
        ("Thermal compensation", f"on · reference {tri.thermal.T0:.0f} °C · camera {res.temps['camera_scan']:.1f} °C "
                                 f"during scan"),
        ("Head geometry", f"R0 {cal.geometry.R0:.3f} mm, h {cal.geometry.h:.2f} mm (ring Ø{cal.ring_D20:.0f}, "
                          f"fit rms {1000 * cal.ring_rms:.0f} µm)"),
        ("Ring checks", ring_txt),
        ("Height reference", f"{res.height.source} (scale {res.height.B:.4f}), anchored at BDC and at the stop: "
                             f"{res.height.stop_source}"),
        ("Orientation", res.yaw_source),
        ("Coverage (non-port bands)", f"min {cov:.0f}% of 1° sectors, largest gap {gaps:.0f}°"),
        ("Outliers", f"{res.n_outliers:,} marked and excluded ({100 * res.n_outliers / max(res.n_points, 1):.2f}%)"),
        ("Flagged bands", f"{len(flagged)} of {len(res.profile)} (see appendix)"),
        ("Crush safety", ("no warnings" if not se else ", ".join(f"{s[1]} at {s[2]:.0f} mm" for s in se[:4]))
         + f" · BLE status {radio.get('received', '?')}/{radio.get('sent', '?')} packets"),
    ], [40 * mm, CW - 40 * mm])
    story += [PageBreak(), Paragraph("Quality and traceability", S["h2"]), qa, Spacer(1, 4 * mm)]
    budget = [["Uncertainty contribution (diameter)", "u (mm)"],
              ["Point noise averaged over a band (Fourier covariance, per band)",
               f"≤ {max(max(b.stats.u_D_FA, b.stats.u_D_PS) for b in allb):.4f}"],
              ["Systematic: calibration, thermal model, encoder, height (simulation-validated)",
               f"{res.meta.get('u_sys', 0.0):.4f}"],
              ["Expanded U (k = 2), worst band", f"{U:.3f}"]]
    story += [_table(budget, [CW - 30 * mm, 30 * mm]), Spacer(1, 4 * mm)]
    story += [Paragraph("Per-position fit quality", S["h3"])]
    qrows = [["Depth mm", "Points", "Outliers", "Cover %", "Max gap °", "Tilt x/y °", "Centre offset mm",
              "σ_r mm", "Ellipse Δ mm", "Flags"]]
    for b in res.positions:
        qrows.append([f"{b.depth:.0f}", f"{b.n_points:,}", f"{b.n_outliers}", f"{b.coverage_pct:.0f}",
                      f"{b.max_gap_deg:.0f}", f"{b.tilt_deg[0]:.2f} / {b.tilt_deg[1]:.2f}",
                      f"{np.hypot(*b.centre_xy):.1f}", f"{b.sigma_r:.3f}", _f(b.ellipse_dev, 3),
                      Paragraph(_short_flags(b), _ps("fq", 6.2, color=INK2, lead=7.2))])
    qw = [16, 17, 15, 14, 15, 24, 22, 15, 18]
    qw = [w_ * mm for w_ in qw]
    qw.append(CW - sum(qw))
    story += [_table(qrows, qw, font_size=6.8), Spacer(1, 4 * mm), Paragraph("Scan timeline", S["h3"])]
    tl = [["Time s", "Event"]]
    for e_ in res.meta.get("events", []):
        if e_[1] in ("state", "safety"):
            tl.append([f"{e_[0]:.1f}", e_[2]])
    story += [_table(tl, [22 * mm, CW - 22 * mm], align_right_from=None, font_size=6.8), Spacer(1, 2 * mm),
              Paragraph("Ellipse cross-check: a direct least-squares ellipse is fitted to every band; where its major "
                        "and minor diameters differ from the Fourier D max / D min by more than 0.05 mm the band is "
                        "flagged as non-elliptical (lobed or locally worn cross-section).", S["tiny"])]

    # ------------------------------------------------------------------ page 5: method & sign-off
    method = [
        "<b>Principle.</b> A 405 nm laser line is projected on the liner wall and observed by a camera at a fixed "
        "baseline. In every image row the sub-pixel position of the line gives the distance to the wall "
        "(laser-line triangulation, z = a/(u − u0) + b per row). The head sweeps ±190° while the turning gear "
        "raises the piston from BDC, so one stroke covers the whole wall.",
        "<b>Geometry.</b> Points are placed with the calibrated arm offset R0, beam offset h and the ray "
        "elevation; each 30 mm band is fitted with a local cylinder (removes device tilt and off-centre placement), "
        "a circle about the band axis and an order-8 Fourier radius profile. D(α) = ρ(α) + ρ(α+180°) across the "
        "fitted centre — never through the device axis.",
        "<b>Definitions.</b> D FA = diameter fore–aft, D PS = diameter port–starboard, ovality = |D PS − D FA| "
        "(UMA definition). D max / D min are the largest / smallest diameter over all directions, with their "
        "angles; max − min is reported separately. Depths are below the liner top.",
        "<b>Height.</b> Crown position from the upward laser distance module, scaled between BDC and the stop "
        "position (liner-top edge seen by the scanner when it is in view, cross-checked with the flywheel reading).",
        "<b>Safety.</b> Planned stop crank angle (primary) and two upward multizone ToF sensors with buzzer "
        "(secondary). Safety does not depend on the radio link.",
    ] + (extra_notes or [])
    story += [PageBreak(), Paragraph("Method and definitions", S["h2"])]
    story += [Paragraph(m, S["body"]) for m in method]
    story += [Spacer(1, 4 * mm), Paragraph("Remarks", S["h2"]),
              _panel([Paragraph("…", S["body"]), Spacer(1, 18 * mm)], CW), Spacer(1, 6 * mm),
              Paragraph("Sign-off", S["h2"])]
    sig = [["Surveyor (UMA Marine)", "Chief Engineer", "Superintendent / Class"],
           ["\n\n\nName / signature / date", "\n\n\nName / signature / date", "\n\n\nName / signature / date"]]
    st = _table(sig, [CW / 3] * 3, zebra=False, align_right_from=None,
                extra=[("BOX", (0, 1), (-1, -1), 0.6, colors.HexColor(RULE)),
                       ("INNERGRID", (0, 1), (-1, -1), 0.6, colors.HexColor(RULE)),
                       ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor(MUTED)), ("VALIGN", (0, 1), (-1, -1), "BOTTOM"),
                       ("FONT", (0, 1), (-1, -1), FONT, 6.5)])
    story += [st]

    # ------------------------------------------------------------------ appendix: all bands
    story += [PageBreak(), Paragraph("Appendix A — every 30 mm band", S["h2"])]
    hdr = ["Depth", "D FA", "D PS", "Ovality", "D max", "°", "D min", "°", "max−min", "Points", "Outl.", "Cover %",
           "U(k=2)", "Flags"]
    rows = [hdr]
    for b in res.profile:
        st_ = b.stats
        fl = _short_flags(b)
        rows.append([f"{b.depth:.0f}", f"{st_.D_FA:.2f}", f"{st_.D_PS:.2f}", f"{st_.ovality:.2f}", f"{st_.D_max:.2f}",
                     f"{st_.D_max_angle_deg:.0f}", f"{st_.D_min:.2f}", f"{st_.D_min_angle_deg:.0f}",
                     f"{st_.max_minus_min:.2f}", f"{b.n_points:,}", f"{b.n_outliers}", f"{b.coverage_pct:.0f}",
                     f"{b.U_k2:.3f}", Paragraph(fl, _ps("fl", 5.8, color=INK2, lead=6.8))])
    cw2 = [11, 14, 14, 12, 14, 7, 14, 7, 12, 14, 9, 11, 12]
    cw2 = [c * mm for c in cw2]
    cw2.append(CW - sum(cw2))
    story += [_table(rows, cw2, font_size=6.6)]
    doc.build(story)
    return out_path


__all__ = ["build_report", "fig_profile", "fig_map", "fig_sections", "fig_3d", "KeepTogether", "os"]
