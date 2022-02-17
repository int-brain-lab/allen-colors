# Imports.

import numpy as np
from IPython.core.display import display, HTML
from ipywidgets import interact
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import colorio
from colorio.cs import ColorCoordinates, HSV, OKLAB, SRGB1, XYZ100

from ibllib.atlas.regions import BrainRegions, FILE_BERYL

# Config.
plt.rcParams["figure.dpi"] = 100
plt.rcParams["axes.grid"] = False

# Brain regions.
br = BrainRegions()
beryl = np.sort(np.load(FILE_BERYL))
kept = br.ancestors(beryl)['id']
beryl_ids = br.id[np.isin(br.id, beryl)]

# Constants.
root = 997  # 0
basic = 8  # 1
cerebrum = 567  # 2
cortex = 688  # 3
isocortex = 315  # 4
nuclei = 623  # 3
brainstem = 343  # 2
interbrain = 1129  # 3
midbrain = 313  # 3
hindbrain = 1065  # 3
cerebellum = 512  # 2
thalamus = 549  # 4
hypothalamus = 1097  # 4

# HTML code generation.
js = '''
function expand() {
    $("details").attr("open", true);
}

function collapse(level) {
    expand();
    $(".level-" + level).attr("open", false);
}

function filter(inp) {
    var s = inp.val();
    console.log(s);
    if (!s) {
        $("details > summary").show();
        return;
    }

    $("details > summary").filter(function (i, el) {
        //console.log(i, el);
        var t = $(el).find(".region-label").text().toLowerCase();
        return t.indexOf(s.toLowerCase()) < 0;
    }).hide();
}

// A $( document ).ready() block.
$(document).ready(function() {
    var el = $('#search');
    el.on('input', function(e){
        filter(el);
    });
})

'''


css = '''
.region-label:hover {color: #000 !important;}
'''


html = f'''
    <html>
    <head><title>Brain region colors</title></head>
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
    <script>
    {js}
    </script>
    <style>
    {css}
    </style>
    <body>
    <div>
    This is a first attempt at providing a modified color map for the Allen Atlas brain regions, for which many regions have identical colors.
    Notes:
    <ul>
    <li>This list shows all Beryl brain regions and their ancestors.</li>
    <li>Click on a region to collapse/extend its descendents.</li>
    <li>For each region, two colors are shown: the original Allen Atlas color, and the optionally modified one.</li>
    <li>For each region, the name, region id, acronym, tree level are shown.</li>
    <li>The name is in bold if the region is a Beryl region.</li>
    <li>Regions with a modified color are highlighted in yellow.</li>
    <li>A modified color is proposed for a region if all of its siblings have the same color.</li>
    <li>A color variant is computed by adding a small random (normal) perturbation to the H, S, V components.</li>
    <li>Required improvements: find a smarter algorithm for generating visually distinct color variations.</li>
    </ul>
    </div>
    <div style="margin: 20px;"><input id="search" type="text" placeholder="search"></div>
    <div style="margin: 20px;">
    <button onclick="collapse(2);">Collapse to L2</button>
    <button onclick="collapse(3);">Collapse to L3</button>
    <button onclick="collapse(4);">Collapse to L4</button>
    <button onclick="collapse(5);">Collapse to L5</button>
    <button onclick="collapse(6);">Collapse to L6</button>
    <button onclick="expand();">Expand all</button>
    </div>
    %s
    </body>
    </html>
    '''


def color_rectangle(level, rgb, is_custom=False, color='#333'):
    rgb8 = float_to_rgb8(rgb)
    return (
        f'<div style="'
        f'width: 100px; height: 30px; '
        f'background-color: rgb{rgb8}; '
        f'margin: 5px 10px 0 {level * 30}px; '
        f'padding: 2px 0 0 5px; '
        f'font-size: .75em; '
        f'border: 1px solid #aaa; '
        f'color: {color}; '
        f'">{"#%02x%02x%02x" % rgb8}</div>\n')


def region_label(level=None, name=None, id=None, acronym=None, is_custom=None, inberyl=None):
    return (
        f'<div class="region-label" style="padding-top: 7px; '
        f'font-weight: {"bold" if inberyl else "normal"}; '
        f'color: {"#555" if inberyl else "#999"}; '
        f'background-color: {"none" if not is_custom else "#ffd"}; '
        f'">L{int(level)} {name} (#{id}, {acronym})</div>\n')


def lines(rgb):
    rgb8 = float_to_rgb8(rgb)
    return (
        f'<div style="margin-left: 20px; margin-top: 5px;">'
        # darkbg
        f'<div style="width: 150px; padding: 5px; background-color: #fff;">'
        f'<div style=" height: 6px; background-color: rgb{rgb8};"></div>'
        f'</div>'
        # white bg
        f'<div style="width: 150px; padding: 4px; background-color: #000;">'
        f'<div style=" height: 6px; background-color: rgb{rgb8};"></div>'
        f'</div>'
        f'</div>\n')


def write_html(path, contents):
    with open(path, "w") as f:
        f.write(html % contents)


# Functions.
def r2h(rgb_float):
    # assume [0, 1]
    return mcolors.rgb_to_hsv(rgb_float)


def h2r(hsv_float):
    # assume [0, 1]
    return mcolors.hsv_to_rgb(hsv_float)


def float_to_rgb8(rgb):
    c = np.array(rgb)
    assert np.all((0 <= c) & (c <= 1))
    return tuple(map(int, (c * 255).astype(np.uint8)))


def oklab_to_xyz(L, a, b):
    M1 = np.array([
        +0.8189330101,
        +0.0329845436,
        +0.0482003018,
        +0.3618667424,
        +0.9293118715,
        +0.2643662691,
        -0.1288597137,
        +0.0361456387,
        +0.6338517070,
    ]).reshape((3, 3)).T

    M2 = np.array([
        +0.2104542553,
        +1.9779984951,
        +0.0259040371,
        +0.7936177850,
        -2.4285922050,
        +0.7827717662,
        -0.0040720468,
        +0.4505937099,
        -0.8086757660,
    ]).reshape((3, 3)).T

    Lab = np.c_[L, a, b].T
    lmsp = np.linalg.inv(M2) @ Lab
    lms = lmsp ** 3
    XYZ = np.linalg.inv(M1) @ lms
    return XYZ


def xyz_to_rgb(xyz):
    M = np.array([
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ])
    return M @ xyz


def oklab_to_rgb(L, a, b):
    xyz = oklab_to_xyz(L, a, b)
    return xyz_to_rgb(xyz)


def get_color(id):
    idx = np.nonzero(br.id == id)[0]
    return br.rgb[idx] / 255.0


def children(ids):
    return br.id[np.isin(br.parent, np.array(ids))]


def children_colors(rid, restrict_to_beryl=None):
    cid = children(rid)
    # print(br.name[np.isin(br.id, did)])
    if restrict_to_beryl:
        ids = cid[np.nonzero(np.isin(cid, beryl))[0]]
    else:
        ids = cid
    return ids, br.rgb[np.isin(br.id, ids)] / 255.0


def make_gradient(L0, C0, n):
    L = L0 * np.ones(n)
    C = C0 * np.ones(n)
    h = np.linspace(-np.pi, np.pi, n)
    a = C * np.cos(h)
    b = C * np.sin(h)
    return oklab_to_rgb(L, a, b).T


def variants(rgb, hstd=0, sstd=0, vstd=0):
    """Generate variations of a set of colors."""
    hsv = r2h(rgb)
    n = len(hsv)
    shape = hsv.shape
    hsv[:, 0] += np.random.normal(size=n, loc=0, scale=hstd)
    hsv[:, 0] %= 1
    hsv[:, 1] += np.random.normal(size=n, loc=0, scale=sstd)
    hsv[:, 2] += np.random.normal(size=n, loc=0, scale=vstd)
    hsv = np.clip(hsv, 0, 1)
    return h2r(hsv)


def make_variants(id, hstd=None, sstd=None, vstd=None):
    """Generate color variants for all children of a given region."""
    # First, we get the original children colors.
    ids, colors = children_colors(id, restrict_to_beryl=False)
    if len(ids) == 0:
        return {}
    # We modify colors only if all children colors are identical.
    if np.all(np.std(colors, axis=0) < 1e-10):
        colors = variants(colors, hstd=hstd, sstd=sstd, vstd=vstd)
    custom = {id: tuple(c.ravel()) for id, c in zip(ids, colors)}
    return custom


def make_variants_recursive(id, hstd=None, sstd=None, vstd=None, decrease_coef=None):
    """Recursively generate color variants for all descendants of a given region."""
    custom = make_variants(
        id, hstd=hstd, sstd=sstd, vstd=vstd)
    for child, child_color in custom.copy().items():
        custom.update(make_variants_recursive(
            child,
            hstd=hstd / decrease_coef,
            sstd=sstd / decrease_coef,
            vstd=vstd / decrease_coef,
            decrease_coef=decrease_coef))
    return custom


def custom_variants():
    hstd = .025
    sstd = .03
    vstd = .02
    decrease_coef = 1.01

    custom = make_variants_recursive(
        root, hstd=hstd, sstd=sstd, vstd=vstd, decrease_coef=decrease_coef)
    return custom


def custom_gradient():
    L0 = .85
    C0 = .1
    n = len(beryl)
    grad = make_gradient(L0, C0, n)
    grad = np.clip(grad, 0, 1)
    custom = {id: grad[(-50 + i) % n] for i, id in enumerate(beryl_ids)}
    return custom


def generate_html(ids, recursive=False, restrict_to_beryl=False, max_level=10, custom=None):
    if recursive:
        cids = br.descendants(ids)['id']
    else:
        cids = children(ids)
    idx = np.isin(br.id, cids) & (br.id >= 0) & np.isin(
        br.id, kept) & (br.level <= max_level)
    idx |= np.isin(br.id, ids)
    ids = br.id[idx]
    names = br.name[idx]
    acronyms = br.acronym[idx]
    colors = br.rgb[idx] / 255.0
    levels = br.level[idx].astype(np.int64)
    inberyls = np.isin(ids, beryl)
    l0 = levels.min()
    last_level = -1
    is_custom = False

    assert len(ids) == len(names) == len(acronyms) == len(
        colors) == len(levels) == len(inberyls)
    s = ''
    for id, name, acronym, color, level, inberyl in zip(ids, names, acronyms, colors, levels, inberyls):
        if name == 'fiber tracts':
            break
        if restrict_to_beryl and not recursive and level == l0 + 1 and not inberyl:
            continue
        rgb = tuple(color)

        if level == last_level:
            s += '</details>\n'
        elif level < last_level:
            s += ('</details>' * (last_level - level + 1))

        s += (
            f'<details open="true" id="area-{id}" class="level-{level}">\n'
            f'<summary style="list-style: none; cursor: pointer;">\n'
            f'<div style="display: flex;">\n')

        # Color
        s += color_rectangle(level - l0, rgb)

        # Optional custom color
        if custom:
            custom_color = custom.get(id, rgb)
            is_custom = np.abs(np.array(custom_color) -
                               np.array(rgb)).max() > 1e-10
            s += color_rectangle(0, (1, 1, 1) if not is_custom else custom_color,
                                 is_custom=is_custom, color='#fff' if not is_custom else '#333')

        # Label
        s += region_label(
            level=level, name=name, id=id,
            acronym=acronym, is_custom=is_custom, inberyl=inberyl)

        # Lines.
        if is_custom:
            s += lines(custom_color)

        s += (
            f'</div>\n'
            f'</summary>\n')

        last_level = level
    s += ("</details>" * (level))
    return s


if __name__ == '__main__':
    write_html(
        "docs/index.html",
        generate_html(root, recursive=True, custom=custom_gradient(), max_level=7))
