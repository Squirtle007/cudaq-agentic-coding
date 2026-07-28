"""Helper functions for the CUDA-Q QAOA map-coloring tutorial (Steps 0-4).

**Read-only**

`data/taiwan_map_xy.json` carries everything the notebooks need about the map:
for each of the 19 main-island counties a simplified outline polygon, a centroid,
its region (4 regions, notebooks 00-01) and its zone (9 zones, notebooks 02-04),
plus the region and zone adjacency lists. The 3 offshore counties have no land borders and are left out, and every
adjacency was computed from the real county borders and verified.

Needs the standard library + matplotlib + networkx + cudaq (`cudaq` is imported
inside qubo_to_spin, not at the top, so a notebook can still set CUDAQ_*
environment variables before its own `import cudaq`).
"""
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import networkx as nx

# Fixed drawing style, and fixed node orders (the order fixes each qubit number).
PALETTE = ["#76B900", "#FAC200", "#0071C5", "#5D1682"]      # colors 0, 1, 2, 3
POLY_FILL, BOUNDARY_COLOR, EDGE_COLOR = "#CDCDCD", "#5E5E5E", "black"
REGION_ORDER = ["North", "Central", "South", "East"]
ZONE_ORDER = ["Taipei", "Keelung", "Taoyuan-Hsinchu-Miaoli", "Yilan",
              "Taichung-Changhua-Nantou", "Yunlin-Chiayi", "Tainan", "South", "East"]
DATA_FILE = Path(__file__).resolve().parent / "data" / "taiwan_map_xy.json"


# ---------------------------------------------------------------------------
# 1. The map file and the two graphs stored in it
# ---------------------------------------------------------------------------
def load_map():
    """-> {"counties": {name: {polygon, centroid, region, zone}},
           "region_edges": [5 pairs], "zone_edges": [14 pairs]}"""
    return json.loads(DATA_FILE.read_text())


def members_of(map_data, level):
    """Group the county names by "region" or "zone" -> {group: [county, ...]}."""
    groups = {}
    for name, county in map_data["counties"].items():
        group = county[level]
        if group not in groups:
            groups[group] = []
        groups[group].append(name)
    return groups


def region_graph(map_data):
    g = nx.Graph()
    g.add_nodes_from(REGION_ORDER)
    g.add_edges_from(map_data["region_edges"])
    return g


def zone_graph(map_data):
    g = nx.Graph()
    g.add_nodes_from(ZONE_ORDER)
    g.add_edges_from(map_data["zone_edges"])
    return g


def group_positions(map_data, level):
    """One (x, y) per region or zone: the average of its members' centroids."""
    positions = {}
    for group, counties in members_of(map_data, level).items():
        sum_x, sum_y = 0.0, 0.0
        for name in counties:
            sum_x = sum_x + map_data["counties"][name]["centroid"][0]
            sum_y = sum_y + map_data["counties"][name]["centroid"][1]
        positions[group] = (sum_x / len(counties), sum_y / len(counties))
    return positions


# ---------------------------------------------------------------------------
# 2. Map drawing (plain matplotlib; same style in every notebook)
# ---------------------------------------------------------------------------
def plot_map(map_data, unit_colors=None, node_pos=None, edges=None, node_colors=None,
             unit_labels=False, color_legend=False, title="", figsize=(6.2, 8.0)):
    """Draw the map, optionally with a graph and a coloring on top.
    Style rules: county polygons #CDCDCD, boundaries #5E5E5E, graph edges black,
    coloring uses PALETTE. `unit_colors` maps county names to fill colors."""
    fig, ax = plt.subplots(figsize=figsize)
    for name, county in map_data["counties"].items():
        xs, ys = [], []
        for point in county["polygon"]:
            xs.append(point[0])
            ys.append(point[1])
        fill = POLY_FILL
        if unit_colors is not None and name in unit_colors:
            fill = unit_colors[name]
        ax.fill(xs, ys, facecolor=fill, edgecolor=BOUNDARY_COLOR, linewidth=0.8, zorder=1)
        if unit_labels:
            short = name.replace(" County", "").replace(" City", "")
            ax.annotate(short, county["centroid"], ha="center", fontsize=6,
                        color="#404040", zorder=5)
    handles = [Line2D([0], [0], color=BOUNDARY_COLOR, lw=1, label="geographic boundary")]
    if node_pos is not None:                     # the graph over the map
        if edges is not None:
            for a, b in edges:
                ax.plot([node_pos[a][0], node_pos[b][0]], [node_pos[a][1], node_pos[b][1]],
                        color=EDGE_COLOR, lw=2.5, zorder=3)
        for name, (x, y) in node_pos.items():
            face = "white"
            if node_colors is not None and name in node_colors:
                face = node_colors[name]
            ax.scatter([x], [y], s=750, c=[face], edgecolors="black", linewidths=1.6, zorder=4)
            ax.annotate(name, (x, y), xytext=(12, 12), textcoords="offset points",
                        fontsize=8, fontweight="bold", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.85))
        handles.append(Line2D([0], [0], color=EDGE_COLOR, lw=2.5, label="graph adjacency edge"))
        handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
                              markeredgecolor="black", markersize=10, label="graph node"))
    if color_legend:
        for c in range(len(PALETTE)):
            handles.append(mpatches.Patch(facecolor=PALETTE[c], edgecolor="black",
                                          label=f"color {c}  {PALETTE[c]}"))
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel("TWD97 / TM2 x (m)", fontsize=8)
    ax.set_ylabel("TWD97 / TM2 y (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(handles=handles, loc="lower right", fontsize=7)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 3. QUBO -> spin Hamiltonian
# ---------------------------------------------------------------------------
def q_index(v, c, nodes, n_colors):
    """Qubit number of the binary variable x[v, c]."""
    return n_colors * nodes.index(v) + c


def build_qubo(nodes, edges, n_colors, A, B):
    """H = A*sum_v (sum_c x_vc - 1)^2  +  B*sum_(u,v in E) sum_c x_uc*x_vc.
    Returns (coeffs, offset): coeffs[(p, q)] is the weight of x_p*x_q,
    and (p, p) means the linear term (because x*x = x for binaries)."""
    coeffs, offset = {}, 0.0
    for v in nodes:                       # one-hot penalty, multiplied out
        offset = offset + A
        for c in range(n_colors):
            p = q_index(v, c, nodes, n_colors)
            coeffs[(p, p)] = coeffs.get((p, p), 0.0) - A
            for c2 in range(c + 1, n_colors):
                p2 = q_index(v, c2, nodes, n_colors)
                coeffs[(p, p2)] = coeffs.get((p, p2), 0.0) + 2 * A
    for u, v in edges:                    # adjacency penalty
        for c in range(n_colors):
            p1 = q_index(u, c, nodes, n_colors)
            p2 = q_index(v, c, nodes, n_colors)
            if p1 > p2:
                p1, p2 = p2, p1
            coeffs[(p1, p2)] = coeffs.get((p1, p2), 0.0) + B
    return coeffs, offset


def qubo_energy(bits, coeffs, offset):
    """Energy of one assignment; `bits` is a list of 0s and 1s (index = qubit)."""
    energy = offset
    for (p, q), w in coeffs.items():
        if p == q:
            energy = energy + w * bits[p]
        else:
            energy = energy + w * bits[p] * bits[q]
    return energy


def qubo_to_spin(coeffs, offset):
    """Substitute x = (1 - Z)/2. Returns (H, terms, spin_offset): H is a CUDA-Q
    SpinOperator, terms = {"z": {q: w}, "zz": {(p, q): w}}, and spin_offset is
    the constant to add back to every measured energy."""
    from cudaq import spin
    hz, hzz, spin_offset = {}, {}, offset
    for (p, q), w in coeffs.items():
        if p == q:                                     # w*x   -> w/2 - (w/2) Z
            spin_offset = spin_offset + w / 2
            hz[p] = hz.get(p, 0.0) - w / 2
        else:                                          # w*x*x -> w/4 (1 - Z - Z + ZZ)
            spin_offset = spin_offset + w / 4
            hz[p] = hz.get(p, 0.0) - w / 4
            hz[q] = hz.get(q, 0.0) - w / 4
            hzz[(p, q)] = hzz.get((p, q), 0.0) + w / 4
    terms = {"z": {}, "zz": {}}
    for p, w in hz.items():                            # keep only non-zero terms
        if abs(w) > 1e-12:
            terms["z"][p] = w
    for pair, w in hzz.items():
        if abs(w) > 1e-12:
            terms["zz"][pair] = w
    H = None
    for p in sorted(terms["z"]):
        H = terms["z"][p] * spin.z(p) if H is None else H + terms["z"][p] * spin.z(p)
    for p, q in sorted(terms["zz"]):
        H = H + terms["zz"][(p, q)] * spin.z(p) * spin.z(q)
    return H, terms, spin_offset


def terms_to_lists(terms):
    """Flatten the spin terms into plain lists (CUDA-Q kernels take lists)."""
    zq, zc, zz1, zz2, zzc = [], [], [], [], []
    for p in sorted(terms["z"]):
        zq.append(p)
        zc.append(terms["z"][p])
    for p, q in sorted(terms["zz"]):
        zz1.append(p)
        zz2.append(q)
        zzc.append(terms["zz"][(p, q)])
    return zq, zc, zz1, zz2, zzc


# ---------------------------------------------------------------------------
# 4. Reading sampled bitstrings
# ---------------------------------------------------------------------------
def decode(bitstring, nodes, n_colors):
    """Bitstring -> {node: color}. A node gets None if it has zero or several
    colors switched on. Character q of a CUDA-Q bitstring is qubit q."""
    assignment = {}
    for v in nodes:
        found = []
        for c in range(n_colors):
            if bitstring[q_index(v, c, nodes, n_colors)] == "1":
                found.append(c)
        assignment[v] = found[0] if len(found) == 1 else None
    return assignment


def is_one_hot(assignment):
    return None not in assignment.values()   # every node has exactly one color


def is_proper(assignment, edges):
    for u, v in edges:
        if assignment[u] is not None and assignment[u] == assignment[v]:
            return False
    return True


def is_valid(bitstring, nodes, n_colors, edges):
    assignment = decode(bitstring, nodes, n_colors)
    return is_one_hot(assignment) and is_proper(assignment, edges)


def valid_rate(counts, nodes, n_colors, edges):
    """Fraction of shots whose bitstring is a valid coloring."""
    total, good = 0, 0
    for bitstring, n in counts.items():
        total = total + n
        if is_valid(bitstring, nodes, n_colors, edges):
            good = good + n
    return good / total if total > 0 else 0.0


def summarize_samples(counts, nodes, n_colors, edges, coeffs, offset):
    """Print the valid-sample rate and the best (most sampled) valid coloring.
    Returns a small dict with the same information."""
    total, valid = 0, {}
    for bitstring, n in counts.items():
        total = total + n
        if is_valid(bitstring, nodes, n_colors, edges):
            valid[bitstring] = n
    n_valid = sum(valid.values())
    print(f"valid colorings among raw samples: {n_valid:,} / {total:,} shots "
          f"= {n_valid / total:.4%}  ({len(valid)} distinct bitstrings)")
    if len(valid) == 0:
        return {"valid_shots": 0, "valid_rate": 0.0, "best": None}
    best = None                                  # the most frequently sampled one
    for bitstring, n in valid.items():
        if best is None or n > valid[best]:
            best = bitstring
    bits = [int(character) for character in best]
    energy = qubo_energy(bits, coeffs, offset)
    assignment = decode(best, nodes, n_colors)
    print(f"best valid coloring: '{best}' (sampled {valid[best]} times, energy {energy:.1f})")
    print("   " + ", ".join(f"{v}={c}" for v, c in assignment.items()))
    return {"valid_shots": n_valid, "valid_rate": n_valid / total, "best": best,
            "best_energy": energy, "assignment": assignment}


# ---------------------------------------------------------------------------
# 5. Timing (timings vary from run to run — warm up first)
# ---------------------------------------------------------------------------
def time_it(fn, warmup=1, repeat=3):
    """Average seconds per call of fn(), measured after `warmup` untimed calls."""
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - start) / repeat