"""One place for the figure style, so that every figure is legible.

Journal figures get reduced, and text that looks adequate on screen at 100 per
cent is unreadable at column width. Nothing here goes below 10 point. Anything
that wants to be smaller than that belongs in the caption instead of on the
figure, which is also where a reader looks for it.
"""
import matplotlib
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.lines import Line2D

CM = 1 / 2.54
INK, BLU, ACC, GRY = '#1a1a1a', '#1D6BAA', '#B4442E', '#9a9a9a'
LAND, GRID = '#e9e6e1', '#ebebeb'


def apply(base=11.0):
    matplotlib.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
        'font.size': base,
        'axes.labelsize': base + 1.5,
        'axes.titlesize': base + 1.0,
        'xtick.labelsize': base,
        'ytick.labelsize': base,
        'legend.fontsize': base,
        'figure.titlesize': base + 2.0,
        'axes.linewidth': 1.0,
        'xtick.major.width': 1.0, 'ytick.major.width': 1.0,
        'xtick.major.size': 4.0, 'ytick.major.size': 4.0,
        'pdf.fonttype': 42, 'savefig.dpi': 400,
        'axes.spines.top': False, 'axes.spines.right': False,
    })


PLACED_CM = 16.01      # the width the manuscript places a full-width figure at
FLOOR_PT = 8.0         # smallest readable size on the printed page


def _placed(fig):
    """Every visible piece of text with the box it actually occupies.

    This has to happen here rather than on the saved file, because a label drawn
    with a halo reaches the PDF as vector outlines rather than as text and is
    invisible to anything reading the file afterwards. Those are the small
    annotations most likely to be too small or to sit on top of something else,
    so they are exactly the ones that must be measured before saving.
    """
    fig.canvas.draw()                       # extents need a renderer
    r = fig.canvas.get_renderer()
    items = list(fig.texts)
    for lg in getattr(fig, 'legends', []):
        items += lg.get_texts()
    for ax in fig.get_axes():
        items += [ax.title, ax.xaxis.label, ax.yaxis.label] + list(ax.texts)
        # Tick labels are taken by location rather than from get_xticklabels: an
        # axis keeps a label artist for every tick the locator proposed, marked
        # visible, including ones outside the view that are never drawn, and
        # measuring those reports collisions that are not on the figure.
        for _ax, _lim, _loc in ((ax.xaxis, ax.get_xlim(), ax.get_xticks()),
                                (ax.yaxis, ax.get_ylim(), ax.get_yticks())):
            _a, _b = sorted(_lim)
            items += [tk.label1 for tk, v in zip(_ax.get_major_ticks(), _loc)
                      if _a <= v <= _b]
        lg = ax.get_legend()
        if lg is not None:
            items += lg.get_texts()
    out = []
    for t in items:
        if not t.get_visible() or not t.get_text().strip():
            continue
        try:
            bb = t.get_window_extent(renderer=r)
            sz = t.get_fontsize()
        except Exception:
            continue
        if bb.width > 0 and bb.height > 0:
            out.append((t, bb, sz))
    return out


def _seg_hits_rect(p0, p1, r, pad):
    """Does the segment p0-p1 come within `pad` of, or cross, the rectangle r?"""
    x0, y0, x1, y1 = r[0] - pad, r[1] - pad, r[2] + pad, r[3] + pad
    ax_, ay = p0
    bx, by = p1
    if (max(ax_, bx) < x0 or min(ax_, bx) > x1
            or max(ay, by) < y0 or min(ay, by) > y1):
        return False
    if x0 <= ax_ <= x1 and y0 <= ay <= y1:
        return True
    if x0 <= bx <= x1 and y0 <= by <= y1:
        return True
    dx, dy = bx - ax_, by - ay
    t0, t1 = 0.0, 1.0
    for pp, qq in ((-dx, ax_ - x0), (dx, x1 - ax_), (-dy, ay - y0), (dy, y1 - ay)):
        if pp == 0:
            if qq < 0:
                return False
            continue
        t = qq / pp
        if pp < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return False
    return True


def _drawn(fig):
    """Points and polylines actually inked by the plot, in display coordinates.

    Lines are kept as polylines rather than as one bounding box: the box of a long
    diagonal covers a large empty rectangle and would report collisions that a reader
    never sees. Markers are kept as points carrying their own radius.
    """
    segs, pts = [], []
    # A legend's own handles sit beside its own labels by construction, so a check
    # that inks them reports every legend as a text-on-object collision. figS1
    # failed the retrace on exactly that: the offending artist was the legend's
    # marker lying inside its own label's box. Legend descendants are not data.
    _in_legend = set()

    def _mark(a):
        _in_legend.add(id(a))
        for ch in getattr(a, 'get_children', lambda: [])():
            _mark(ch)

    for _ax in fig.get_axes():
        _lg = _ax.get_legend()
        if _lg is not None:
            _mark(_lg)
    for _lg in getattr(fig, 'legends', []):
        _mark(_lg)

    for ax in fig.get_axes():
        for ln in ax.get_lines():
            if id(ln) in _in_legend:
                continue
            if not ln.get_visible():
                continue
            xy = ln.get_xydata()
            if xy is None or len(xy) == 0:
                continue
            d = ax.transData.transform(xy)
            d = d[np.isfinite(d).all(axis=1)]
            # Only what is actually visible. A line runs on past the axis limits and
            # is clipped when drawn; testing its off-screen vertices reported the
            # curve colliding with a tick label sitting outside the panel.
            ab = ax.get_window_extent()
            vis = ((d[:, 0] >= ab.x0 - 2) & (d[:, 0] <= ab.x1 + 2)
                   & (d[:, 1] >= ab.y0 - 2) & (d[:, 1] <= ab.y1 + 2))
            if ln.get_linestyle() not in (None, 'None', ' ', '') and vis.sum() > 1:
                keep = vis | np.r_[vis[1:], False] | np.r_[False, vis[:-1]]
                segs.append((d[keep], ln))
            if ln.get_marker() not in (None, 'None', ' ', ''):
                r = float(ln.get_markersize()) * fig.dpi / 72.0 / 2.0
                for q in d[vis]:
                    pts.append((q, r, ln))
        for col in ax.collections:
            if not col.get_visible() or id(col) in _in_legend:
                continue
            try:
                off = col.get_offsets()
                d = ax.transData.transform(np.asarray(off, float))
                d = d[np.isfinite(d).all(axis=1)]
            except Exception:
                continue
            # get_sizes belongs to PathCollection; a LineCollection has offsets but no
            # marker size, and asking for one crashed the check and so the whole figure.
            sz = col.get_sizes() if hasattr(col, 'get_sizes') else []
            r = (float(np.sqrt(np.max(sz))) if len(sz) else 6.0) * fig.dpi / 72.0 / 2.0
            ab = ax.get_window_extent()
            for q in d:
                if ab.x0 - 2 <= q[0] <= ab.x1 + 2 and ab.y0 - 2 <= q[1] <= ab.y1 + 2:
                    pts.append((q, r, col))
        for pa in ax.patches:
            if (not pa.get_visible() or pa is getattr(ax, 'patch', None)
                    or id(pa) in _in_legend):
                continue
            try:
                # transform_path, not transform(vertices): under a non-linear
                # transform such as a polar projection the straight line between two
                # transformed corners is not the transformed edge. Joining the corners
                # of a rose bar that way produced chords sweeping clear across the
                # figure and reported collisions with text nowhere near any bar.
                # get_transform() already composes the patch transform with the data
                # transform. Applying get_patch_transform() a second time, as this did
                # until 22 September 2026, doubled a bar's height and width in display
                # space, so the check reported bars colliding with text they did not
                # touch (fig_bands' legend, against a 6 per cent bar drawn as 50).
                pth = pa.get_transform().transform_path(pa.get_path())
                d = np.asarray(pth.vertices, float)
                d = d[np.isfinite(d).all(axis=1)]
            except Exception:
                continue
            if len(d) > 1:
                ab = ax.get_window_extent()
                inside = ((d[:, 0] >= ab.x0 - 1) & (d[:, 0] <= ab.x1 + 1)
                          & (d[:, 1] >= ab.y0 - 1) & (d[:, 1] <= ab.y1 + 1))
                if inside.any():
                    segs.append((d[inside], pa))
    return segs, pts


def legend(ax, *, loc='upper right', grow=0.6, pad=3.0, **kw):
    """A legend guaranteed not to sit on the data, placed by measurement.

    A legend pinned to a corner is correct until the data move. The occupancy panel
    of figure S1 was pinned to the upper right and passed every check until a re-run
    widened the corridors by 60 km and lifted two curves into the box; the figure
    step then failed the whole retrace. Nothing about the figure was wrong, and
    nothing about the data was wrong - the position was a guess that stopped being
    true.

    The requested corner is tried first, so a figure that has room keeps the layout
    its author chose. If it is occupied, the other corners are measured in turn. If
    every corner is occupied, the y axis is grown upward - by at most `grow` of the
    current span - until the requested corner is clear, which keeps the legend where
    the author put it and buys the space rather than moving the box somewhere
    arbitrary. Returns the legend.
    """
    order = [loc] + [x for x in ('upper right', 'upper left', 'lower right',
                                 'lower left', 'center right', 'center left')
                     if x != loc]

    def clear(where):
        lg = ax.legend(loc=where, **kw)
        ax.figure.canvas.draw()
        bb = lg.get_window_extent()
        r = (bb.x0, bb.y0, bb.x1, bb.y1)
        segs, pts = _drawn(ax.figure)
        for d, artist in segs:
            if getattr(artist, 'axes', None) is not ax:
                continue
            for i in range(len(d) - 1):
                if _seg_hits_rect(d[i], d[i + 1], r, pad):
                    return lg, False
        for q, rad, artist in pts:
            if getattr(artist, 'axes', None) is not ax:
                continue
            if (r[0] - rad - pad <= q[0] <= r[2] + rad + pad
                    and r[1] - rad - pad <= q[1] <= r[3] + rad + pad):
                return lg, False
        return lg, True

    for where in order:
        lg, ok = clear(where)
        if ok:
            return lg
        lg.remove()

    lo, hi = ax.get_ylim()
    span = hi - lo
    for step in (0.10, 0.20, 0.30, 0.45, grow):
        if step > grow:
            break
        ax.set_ylim(lo, hi + step * span)
        lg, ok = clear(loc)
        if ok:
            return lg
        lg.remove()
    # Nothing cleared it. Return the legend at the requested place and let check()
    # report the collision rather than silently shipping a worse position.
    return ax.legend(loc=loc, **kw)


def mask(alpha=0.82):
    """Background for a label that must sit on a line it is naming.

    A label over a plotted object is a defect unless it carries something to read
    against. This is that something, and check() recognises it and stops
    complaining, so the two stay in step.
    """
    return dict(facecolor='white', edgecolor='none', alpha=alpha,
                boxstyle='round,pad=0.16')


def check(fig, placed_cm=PLACED_CM, floor=FLOOR_PT, gap=1.5):
    """Fail loudly rather than shipping a figure with unreadable or piled-up text.

    A point size is judged at the width the figure is printed, not the width it
    is drawn. Ten point on a figure 18 cm wide is under nine point in a 16 cm
    column, while the same ten point on a 12 cm figure is over thirteen, so a
    fixed floor on the drawn size passes and fails the wrong figures.
    """
    k = placed_cm / (fig.get_size_inches()[0] * 2.54)
    px = 72.0 / fig.dpi * k                 # display pixels to printed points
    items = _placed(fig)
    small = [(t.get_text()[:34], sz) for t, _, sz in items if sz * k < floor]
    hits = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (ta, ba, _), (tb, bb, _) = items[i], items[j]
            w = min(ba.x1, bb.x1) - max(ba.x0, bb.x0)
            h = min(ba.y1, bb.y1) - max(ba.y0, bb.y0)
            if w * px > gap and h * px > gap:
                hits.append((ta.get_text()[:28], tb.get_text()[:28],
                             min(w, h) * px))
    # A label clipped at the page edge is the one defect a reader cannot work
    # around, and bbox_inches='tight' does not always rescue an overhanging one.
    fw, fh = fig.canvas.get_width_height()
    over = []
    for t, bb, _ in items:
        d = max(-bb.x0, bb.x1 - fw, -bb.y0, bb.y1 - fh)
        if d * px > 1.0:
            over.append((t.get_text()[:40], d * px))
    stray = []
    for _t, bb, _ in items:
        ax = getattr(_t, 'axes', None)
        if ax is None or _t.get_transform() is not ax.transData:
            continue
        ab = ax.get_window_extent()
        d = max(ab.x0 - bb.x1, bb.x0 - ab.x1, ab.y0 - bb.y1, bb.y0 - ab.y1)
        if d * px > 4.0:
            stray.append((_t.get_text()[:40], d * px))
    # Text sitting on a line, a marker or a bar is the defect the eye sees first and
    # the text-against-text test cannot see at all. Figures 4 and 5 shipped with it.
    onobj = []
    haloed = []                 # allowed to sit on a line, but reported
    _segs, _pts = _drawn(fig)
    for t, bb, _ in items:
        if not t.get_text().strip():
            continue
        # A label carrying an opaque background masks whatever it sits on and is
        # legible there by construction; flagging it would train the reader to
        # ignore this check.
        _bp = t.get_bbox_patch()
        if _bp is not None:
            _fc = _bp.get_facecolor()
            if _fc is not None and len(_fc) > 3 and _fc[3] > 0.55:
                continue
        # Legend text inside an opaque framed legend has the same protection: the
        # frame is its background even though the text carries no bbox of its own.
        _lg = None
        for _ax in fig.get_axes():
            _l = _ax.get_legend()
            if _l is not None and t in _l.get_texts():
                _lg = _l
                break
        if _lg is not None and _lg.get_frame_on():
            _f = _lg.get_frame()
            _fc = _f.get_facecolor()
            if _f.get_alpha() in (None, 1.0) and _fc is not None and len(_fc) > 3 \
                    and _fc[3] > 0.55:
                continue
        # A label carrying a stroke halo is legible over a THIN LINE by
        # construction: the halo masks the line within the glyph strokes, which
        # is the standard way to put an annotation on a curve. It is not
        # protection against a filled patch or a marker cloud, so the exemption
        # is granted only against Line2D and the label is listed, never waived
        # silently.
        _haloed = False
        try:
            for _e in (t.get_path_effects() or ()):
                if isinstance(_e, pe.Stroke):
                    _haloed = True
                    break
        except Exception:
            _haloed = False

        r = (bb.x0, bb.y0, bb.x1, bb.y1)
        pad = -gap / px                     # a touch of overlap is not a collision
        for d, art in _segs:
            if any(_seg_hits_rect(d[q], d[q + 1], r, pad) for q in range(len(d) - 1)):
                if _haloed and isinstance(art, Line2D):
                    haloed.append((t.get_text()[:34], type(art).__name__))
                    continue
                onobj.append((t.get_text()[:34], type(art).__name__))
                break
        else:
            for q, rad, art in _pts:
                if (r[0] - rad - pad <= q[0] <= r[2] + rad + pad
                        and r[1] - rad - pad <= q[1] <= r[3] + rad + pad):
                    onobj.append((t.get_text()[:34], type(art).__name__))
                    break
    # A legend's handles are left out of _drawn because they sit beside their own
    # labels by construction. They are still ink, and any other text lying on one is a
    # collision that none of the tests above can see: Figure S1 went out with its panel
    # letter drawn across the handle in the second row of the legend above the panel.
    _r = fig.canvas.get_renderer()
    _g = gap / px                           # the same allowance as for a data line
    _legs = [_a.get_legend() for _a in fig.get_axes() if _a.get_legend() is not None]
    _legs += list(getattr(fig, 'legends', []))
    for _l in _legs:
        _own = {id(x) for x in _l.get_texts()}
        for _h in (getattr(_l, 'legend_handles', None)
                   or getattr(_l, 'legendHandles', None) or []):
            if _h is None or not _h.get_visible():
                continue
            try:
                hb = _h.get_window_extent(_r)
            except Exception:
                continue
            for t, bb, _ in items:
                if id(t) in _own or not t.get_text().strip():
                    continue
                if (hb.x1 > bb.x0 + _g and hb.x0 < bb.x1 - _g
                        and hb.y1 > bb.y0 + _g and hb.y0 < bb.y1 - _g):
                    onobj.append((t.get_text()[:34], 'legend handle'))
    if haloed:
        print('  figstyle: halo allows these labels to sit on a line: '
              + ', '.join(sorted({txt for txt, _ in haloed})))
    if onobj:
        raise SystemExit(
            '\ntext is drawn on top of plotted objects:\n'
            + '\n'.join(f'  {k:<14s} under {txt!r}' for txt, k in onobj))
    if stray:
        raise SystemExit(
            '\ntext placed in data coordinates has landed outside its own '
            'panel:\n'
            + '\n'.join(f'  {d:5.1f} pt away  {txt!r}' for txt, d in stray))
    if over:
        raise SystemExit(
            '\ntext extends past the edge of this figure:\n'
            + '\n'.join(f'  {d:5.1f} pt beyond  {txt!r}' for txt, d in over))
    if hits:
        raise SystemExit(
            '\ntext collides on this figure, printed at %.1f cm:\n' % placed_cm
            + '\n'.join(f'  {d:4.1f} pt  {a!r} / {b!r}' for a, b, d in hits))
    if small:
        raise SystemExit(
            f'\ntext under {floor:.0f} pt once this figure is printed at '
            f'{placed_cm:.1f} cm (drawn at '
            f'{fig.get_size_inches()[0] * 2.54:.1f} cm, so x{k:.2f}):\n'
            + '\n'.join(f'  {sz * k:4.1f} pt printed ({sz:4.1f} drawn)  {txt!r}'
                         for txt, sz in small))
    return True
