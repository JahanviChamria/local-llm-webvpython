"""Headless stand-in for the Web VPython runtime.

Enough of the GlowScript API to exec a sim without a browser: display
objects accept any attributes, vectors do real math, rate() counts steps
and raises StopSim at the budget so infinite animation loops terminate.
Passing eval = the program runs under this stub without an exception
other than StopSim.
"""

import math
import random as _random

MAX_STEPS = 300


class StopSim(Exception):
    """Raised by rate() once the step budget is reached. Counts as a pass."""


class vector:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def __add__(self, o):
        return vector(self.x + o.x, self.y + o.y, self.z + o.z)

    def __sub__(self, o):
        return vector(self.x - o.x, self.y - o.y, self.z - o.z)

    def __mul__(self, s):
        return vector(self.x * s, self.y * s, self.z * s)

    __rmul__ = __mul__

    def __truediv__(self, s):
        return vector(self.x / s, self.y / s, self.z / s)

    def __neg__(self):
        return vector(-self.x, -self.y, -self.z)

    def __eq__(self, o):
        return isinstance(o, vector) and (self.x, self.y, self.z) == (o.x, o.y, o.z)

    def __repr__(self):
        return f"<{self.x}, {self.y}, {self.z}>"

    @property
    def mag(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    @property
    def mag2(self):
        return self.x**2 + self.y**2 + self.z**2

    @property
    def hat(self):
        m = self.mag
        return vector(0, 0, 0) if m == 0 else self / m

    def norm(self):
        return self.hat

    def dot(self, o):
        return self.x * o.x + self.y * o.y + self.z * o.z

    def cross(self, o):
        return vector(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )


def mag(v):
    return v.mag


def mag2(v):
    return v.mag2


def norm(v):
    return v.hat


def hat(v):
    return v.hat


def dot(a, b):
    return a.dot(b)


def cross(a, b):
    return a.cross(b)


class _Thing:
    """Any display object: sphere, box, graph, ... Accepts any attributes."""

    def __init__(self, *args, **kwargs):
        self.pos = vector(0, 0, 0)
        self.axis = vector(1, 0, 0)
        self.color = vector(1, 1, 1)
        self.up = vector(0, 1, 0)
        self.size = vector(1, 1, 1)
        self.visible = True
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getattr__(self, name):  # tolerate reads of never-set attributes
        return None

    def _noop(self, *a, **k):
        return self

    # methods sims commonly call
    plot = append = modify = clear = rotate = delete = clone = _noop


class _Color:
    red = vector(1, 0, 0)
    green = vector(0, 1, 0)
    blue = vector(0, 0, 1)
    yellow = vector(1, 1, 0)
    orange = vector(1, 0.6, 0)
    cyan = vector(0, 1, 1)
    magenta = vector(1, 0, 1)
    purple = vector(0.4, 0, 0.6)
    black = vector(0, 0, 0)
    white = vector(1, 1, 1)
    gray = staticmethod(lambda l: vector(l, l, l))
    rgb_to_hsv = hsv_to_rgb = staticmethod(lambda v: v)


class _Rate:
    def __init__(self):
        self.calls = 0

    def __call__(self, n=60):
        self.calls += 1
        if self.calls >= MAX_STEPS:
            raise StopSim()


def build_globals() -> dict:
    g: dict = {"__builtins__": __builtins__}
    thing_names = [
        "sphere", "box", "cylinder", "cone", "arrow", "helix", "ring",
        "ellipsoid", "pyramid", "curve", "points", "label", "text",
        "compound", "extrusion", "graph", "gcurve", "gdots", "gvbars",
        "ghbars", "canvas", "button", "slider", "menu", "wtext", "winput",
        "checkbox", "radio", "attach_trail", "attach_arrow", "local_light",
        "distant_light",
    ]
    for name in thing_names:
        g[name] = _Thing
    g.update(
        vector=vector, vec=vector, color=_Color, scene=_Thing(),
        rate=_Rate(), mag=mag, mag2=mag2, norm=norm, hat=hat, dot=dot,
        cross=cross, pi=math.pi, sin=math.sin, cos=math.cos, tan=math.tan,
        asin=math.asin, acos=math.acos, atan=math.atan, atan2=math.atan2,
        sqrt=math.sqrt, exp=math.exp, log=math.log, floor=math.floor,
        ceil=math.ceil, radians=math.radians, degrees=math.degrees,
        random=_random.random, factorial=math.factorial,
        arange=lambda a, b=None, s=1: ([a] and list(_frange(0, a, 1)) if b is None else list(_frange(a, b, s))),
        sleep=lambda t: None, get_library=lambda url: None,
    )
    return g


def _frange(a, b, s):
    x = a
    while (s > 0 and x < b) or (s < 0 and x > b):
        yield x
        x += s
