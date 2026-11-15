"""

For each diagram it creates:
  - a PNG image (for the report)
  - a .drawio file (to open and edit in draw.io / app.diagrams.net)

Both come from the SAME list of shapes and arrows below, so they always match.

Run it from the project folder:
    python make_diagrams.py
The images are saved in the learning folder: ~/Projects/Recommendation-system-LEARN/07_dfd/

DFD symbols used:
  rectangle             = external entity (the User)
  circle                = process (a step that changes data)
  open-ended rectangle  = data store (the MovieLens files)
  arrow with a label    = data flow (what data moves where)
"""

import math
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")  # draw to a file, no window
import matplotlib.pyplot as plt

# Where to save the diagrams: the learning folder, so the project folder stays clean
OUT = Path.home() / "Projects" / "Recommendation-system-LEARN" / "07_dfd"
OUT.mkdir(parents=True, exist_ok=True)
SCALE = 10  # 1 diagram unit = 10 pixels in draw.io

INK, PROCESS_FILL, ENTITY_FILL, STORE_FILL = "#1c1917", "#e8f1fc", "#fdf0e8", "#eef8f2"



# The two diagrams: shapes and arrows
# Coordinates: x goes right, y goes UP. (x, y) is the centre of a shape.

def entity(name, x, y, w=12, h=7):
    return {"kind": "entity", "text": name, "x": x, "y": y, "w": w, "h": h}

def process(text, x, y, r=6.2):
    return {"kind": "process", "text": text, "x": x, "y": y, "r": r}

def store(sid, text, x, y, w=26, h=7):
    return {"kind": "store", "id": sid, "text": text, "x": x, "y": y, "w": w, "h": h}

def flow(src, dst, label, points=(), at=None, y=None):
    """An arrow from src to dst. `points` = corners to pass through, `at` = label position,
    `y` = draw it as a straight horizontal arrow at this height."""
    return {"src": src, "dst": dst, "label": label, "points": list(points), "at": at, "y": y}


LEVEL0 = {
    "title": "DFD Level 0 — Context Diagram: Movie Recommender System",
    "size": (100, 40),
    "shapes": {
        "user": entity("User", 7, 20, w=11, h=28),
        "P0": process("0\nMovie\nRecommender\nSystem", 46, 20, r=13),
        "D1": store("D1", "MovieLens Dataset\n(movies.csv, ratings.csv)", 89, 20, w=20, h=16),
    },
    "flows": [
        flow("user", "P0", "movie title", y=30),
        flow("user", "P0", "user ID", y=25),
        flow("P0", "user", "movie title suggestions", y=20),
        flow("P0", "user", "10 similar movies (similarity %)", y=15),
        flow("P0", "user", "10 recommended movies (predicted rating)", y=10),
        flow("D1", "P0", "movies\n(movieId, title, genres)", y=25),
        flow("D1", "P0", "ratings\n(userId, movieId, rating)", y=15),
    ],
}

LEVEL1 = {
    "title": "DFD Level 1 — Inside the Movie Recommender System",
    "size": (104, 68),
    "shapes": {
        "D1": store("D1", "MovieLens Dataset\n(movies.csv, ratings.csv)", 16, 60, w=26, h=7),
        "user": entity("User", 96, 30, w=11, h=8),
        "P1": process("1.0\nLoad movie\ndata", 16, 40),
        "P2": process("2.0\nConvert genres\nto vectors\n(TF-IDF)", 36, 48),
        "P3": process("3.0\nCompute\ncosine\nsimilarity", 55, 48),
        "P4": process("4.0\nSelect top 10\nsimilar movies", 74, 48),
        "P5": process("5.0\nDisplay results\non the website", 74, 24),
        "P6": process("6.0\nPredict\nratings\nwith SVD", 38, 10),
        "P7": process("7.0\nSelect top 10\nrecommen-\ndations", 60, 10),
    },
    "flows": [
        # Content-based branch (cosine similarity)
        flow("D1", "P1", "movies.csv + ratings.csv", at=(17, 52.5)),
        flow("P1", "P2", "movie genres", at=(23.5, 47)),
        flow("P2", "P3", "genre vectors\n(TF-IDF matrix)", at=(45.5, 53.6)),
        flow("user", "P3", "selected movie title", points=[(96, 64), (55, 64)], at=(76, 64)),
        flow("P3", "P4", "similarity scores\n(0 to 1)", at=(64.5, 53.6)),
        flow("P1", "P4", "number of ratings per movie (tie-break)", points=[(68, 37)], at=(45, 38.3)),
        flow("P4", "P5", "top 10 similar\nmovies (JSON)", at=(79.5, 36)),
        flow("P1", "P5", "all movie titles (search-box suggestions)", points=[(24, 24)], at=(45, 24)),
        flow("P5", "user", "10 movie cards", at=(85.5, 29.5)),
        # Collaborative filtering branch (SVD)
        flow("P1", "P6", "ratings\n(userId, movieId,\nrating)", points=[(16, 10)], at=(16, 21)),
        flow("user", "P6", "user ID", points=[(96, 1), (38, 1)], at=(76, 1)),
        flow("P6", "P7", "predicted ratings\nfor unseen movies", at=(49, 6)),
        flow("P7", "P5", "top 10 recommen-\ndations (JSON)", at=(73.5, 15)),
    ],
}


# Geometry helper: where an arrow touches the edge of a shape

def edge_point(shape, toward):
    """Point on the border of `shape` in the direction of the point `toward`."""
    cx, cy = shape["x"], shape["y"]
    dx, dy = toward[0] - cx, toward[1] - cy
    if shape["kind"] == "process":
        d = math.hypot(dx, dy) or 1
        return cx + shape["r"] * dx / d, cy + shape["r"] * dy / d
    hw, hh = shape["w"] / 2, shape["h"] / 2
    t = min(hw / abs(dx) if dx else math.inf, hh / abs(dy) if dy else math.inf)
    return cx + dx * t, cy + dy * t


def edge_x_at(shape, y, side):
    """x of the left (side=-1) or right (side=+1) border of a shape at height y."""
    if shape["kind"] == "process":
        return shape["x"] + side * math.sqrt(max(shape["r"] ** 2 - (y - shape["y"]) ** 2, 0))
    return shape["x"] + side * shape["w"] / 2


def path_of(diagram, f):
    """Full list of points of an arrow: border of src -> corners -> border of dst."""
    s, d = diagram["shapes"][f["src"]], diagram["shapes"][f["dst"]]
    if f["y"] is not None:  # straight horizontal arrow
        side = 1 if d["x"] > s["x"] else -1
        return [(edge_x_at(s, f["y"], side), f["y"]), (edge_x_at(d, f["y"], -side), f["y"])]
    pts = f["points"]
    start = edge_point(s, pts[0] if pts else (d["x"], d["y"]))
    end = edge_point(d, pts[-1] if pts else (s["x"], s["y"]))
    return [start, *pts, end]



# 1) PNG with matplotlib

def draw_png(diagram, filename):
    W, H = diagram["size"]
    fig, ax = plt.subplots(figsize=(W / 6, H / 6))
    ax.set_xlim(0, W)
    ax.set_ylim(-2, H)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    for s in diagram["shapes"].values():
        x, y = s["x"], s["y"]
        if s["kind"] == "process":
            ax.add_patch(plt.Circle((x, y), s["r"], fc=PROCESS_FILL, ec=INK, lw=1.6, zorder=3))
            ax.text(x, y, s["text"], ha="center", va="center", fontsize=9.5, zorder=4, linespacing=1.25)
        elif s["kind"] == "entity":
            ax.add_patch(plt.Rectangle((x - s["w"] / 2, y - s["h"] / 2), s["w"], s["h"],
                                       fc=ENTITY_FILL, ec=INK, lw=1.6, zorder=3))
            ax.text(x, y, s["text"], ha="center", va="center", fontsize=12, weight="bold", zorder=4)
        else:  # data store: open on the right side, with a small "D1" box on the left
            x0, x1, y0, y1 = x - s["w"] / 2, x + s["w"] / 2, y - s["h"] / 2, y + s["h"] / 2
            ax.add_patch(plt.Rectangle((x0, y0), s["w"], s["h"], fc=STORE_FILL, ec="none", zorder=2))
            ax.plot([x1, x0, x0, x1], [y1, y1, y0, y0], color=INK, lw=1.6, zorder=3)
            ax.plot([x0 + 4, x0 + 4], [y0, y1], color=INK, lw=1.6, zorder=3)
            ax.text(x0 + 2, y, s["id"], ha="center", va="center", fontsize=10, weight="bold", zorder=4)
            ax.text(x0 + 4 + (s["w"] - 4) / 2, y, s["text"], ha="center", va="center", fontsize=9.5, zorder=4)

    for f in diagram["flows"]:
        pts = path_of(diagram, f)
        xs, ys = zip(*pts)
        ax.plot(xs[:-1], ys[:-1], color=INK, lw=1.2, zorder=1)
        ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=1,
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2, mutation_scale=14,
                                    shrinkA=0, shrinkB=0))
        if f["at"]:
            lx, ly = f["at"]
        elif f["y"] is not None:  # straight arrow: label just above the line
            lx, ly = (pts[0][0] + pts[1][0]) / 2, f["y"] + (2.2 if "\n" in f["label"] else 1.4)
        else:  # middle of the longest segment
            a, b = max(zip(pts, pts[1:]), key=lambda ab: math.dist(*ab))
            lx, ly = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        ax.text(lx, ly, f["label"], ha="center", va="center", fontsize=8.5, color=INK, zorder=5,
                linespacing=1.2, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"))

    ax.set_title(diagram["title"], fontsize=14, weight="bold", pad=10)
    fig.savefig(OUT / filename, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)



# 2) .drawio file (XML that draw.io understands)
def draw_drawio(diagram, filename):
    W, H = diagram["size"]
    px = lambda x: round(x * SCALE)              # diagram units -> pixels
    py = lambda y: round((H - y) * SCALE)        # draw.io's y axis points DOWN
    html = lambda t: escape(t).replace("\n", "<br>")
    cells = []

    for sid, s in diagram["shapes"].items():
        if s["kind"] == "process":
            r = s["r"]
            style = f"ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor={PROCESS_FILL};strokeColor={INK};strokeWidth=2;fontSize=13;"
            geo = (px(s["x"] - r), py(s["y"] + r), px(2 * r), px(2 * r))
            text = html(s["text"])
        elif s["kind"] == "entity":
            style = f"rounded=0;whiteSpace=wrap;html=1;fillColor={ENTITY_FILL};strokeColor={INK};strokeWidth=2;fontSize=16;fontStyle=1;"
            geo = (px(s["x"] - s["w"] / 2), py(s["y"] + s["h"] / 2), px(s["w"]), px(s["h"]))
            text = html(s["text"])
        else:  # open-ended rectangle (no right side) = data store
            style = (f"shape=partialRectangle;whiteSpace=wrap;html=1;right=0;top=1;bottom=1;left=1;"
                     f"fillColor={STORE_FILL};strokeColor={INK};strokeWidth=2;fontSize=13;")
            geo = (px(s["x"] - s["w"] / 2), py(s["y"] + s["h"] / 2), px(s["w"]), px(s["h"]))
            text = f"<b>{s['id']}</b> | " + html(s["text"])
        x, y, w, h = geo
        cells.append(f'<mxCell id="{sid}" value="{escape(text, {chr(34): "&quot;"})}" style="{style}" vertex="1" parent="1">'
                     f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

    for i, f in enumerate(diagram["flows"]):
        style = (f"endArrow=block;endFill=1;html=1;rounded=0;strokeColor={INK};strokeWidth=1.5;"
                 "fontSize=12;labelBackgroundColor=#ffffff;")
        # Straight arrows: give draw.io their exact start/end so parallel arrows don't overlap
        corners = path_of(diagram, f) if f["y"] is not None else f["points"]
        points = "".join(f'<mxPoint x="{px(a)}" y="{py(b)}"/>' for a, b in corners)
        cells.append(f'<mxCell id="f{i}" value="{escape(html(f["label"]), {chr(34): "&quot;"})}" style="{style}" edge="1" '
                     f'parent="1" source="{f["src"]}" target="{f["dst"]}">'
                     f'<mxGeometry relative="1" as="geometry"><Array as="points">{points}</Array></mxGeometry></mxCell>')

    xml = (f'<mxfile host="app.diagrams.net"><diagram name="{escape(diagram["title"])}">'
           f'<mxGraphModel dx="1200" dy="800" grid="1" gridSize="10" page="1" pageWidth="{px(W) + 40}" '
           f'pageHeight="{px(H) + 40}" background="#ffffff"><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
           + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>")
    (OUT / filename).write_text(xml, encoding="utf-8")


if __name__ == "__main__":
    for diagram, name in [(LEVEL0, "dfd_level0_context"), (LEVEL1, "dfd_level1")]:
        draw_png(diagram, name + ".png")
        draw_drawio(diagram, name + ".drawio")
        print("saved", name + ".png", "and", name + ".drawio")
