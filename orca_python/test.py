# Copyright (c) 2013 Mak Nazecic-Andrlon 
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from __future__ import division

#from pyorca import Agent, get_avoidance_velocity, orca, normalized, perp
from numpy import array, rint, linspace, pi, cos, sin
import pygame

import numpy
from numpy import array, sqrt, copysign, dot
from numpy.linalg import det

import itertools
import random
import sys 
sys.path.append('C:/Users/dell/Desktop/Re_ examples/pyorca.py')
sys.path.append('C:/Users/dell/Desktop/Re_ examples/halfplaneintersect.py')


N_AGENTS = 8
RADIUS = 8.
SPEED = 1

def norm_sq(x):
    return dot(x, x)

def normalized(x):
    l = norm_sq(x)
    assert l > 0, (x, l)
    return x / sqrt(l)

def dist_sq(a, b):
    return norm_sq(b - a)

def perp(a):
    return array((a[1], -a[0]))

def norm_sq(x):
    return dot(x, x)

def norm(x):
    return sqrt(norm_sq(x))

def normalized(x):
    l = norm_sq(x)
    assert l > 0, (x, l)
    return x / sqrt(l)

class Agent(object):
    """A disk-shaped agent."""
    def __init__(self, position, velocity, radius, max_speed, pref_velocity):
        super(Agent, self).__init__()
        self.position = array(position)
        self.velocity = array(velocity)
        self.radius = radius
        self.max_speed = max_speed
        self.pref_velocity = array(pref_velocity)

def orca(agent, colliding_agents, t, dt):
    """Compute ORCA solution for agent. NOTE: velocity must be _instantly_
    changed on tick *edge*, like first-order integration, otherwise the method
    undercompensates and you will still risk colliding."""
    lines = []
    for collider in colliding_agents:
        dv, n = get_avoidance_velocity(agent, collider, t, dt)
        line = Line(agent.velocity + dv / 2, n)
        lines.append(line)
    return halfplane_optimize(lines, agent.pref_velocity), lines

def get_avoidance_velocity(agent, collider, t, dt):
    """Get the smallest relative change in velocity between agent and collider
    that will get them onto the boundary of each other's velocity obstacle
    (VO), and thus avert collision."""

    # This is a summary of the explanation from the AVO paper.
    #
    # The set of all relative velocities that will cause a collision within
    # time tau is called the velocity obstacle (VO). If the relative velocity
    # is outside of the VO, no collision will happen for at least tau time.
    #
    # The VO for two moving disks is a circularly truncated triangle
    # (spherically truncated cone in 3D), with an imaginary apex at the
    # origin. It can be described by a union of disks:
    #
    # Define an open disk centered at p with radius r:
    # D(p, r) := {q | ||q - p|| < r}        (1)
    #
    # Two disks will collide at time t iff ||x + vt|| < r, where x is the
    # displacement, v is the relative velocity, and r is the sum of their
    # radii.
    #
    # Divide by t:  ||x/t + v|| < r/t,
    # Rearrange: ||v - (-x/t)|| < r/t.
    #
    # By (1), this is a disk D(-x/t, r/t), and it is the set of all velocities
    # that will cause a collision at time t.
    #
    # We can now define the VO for time tau as the union of all such disks
    # D(-x/t, r/t) for 0 < t <= tau.
    #
    # Note that the displacement and radius scale _inversely_ proportionally
    # to t, generating a line of disks of increasing radius starting at -x/t.
    # This is what gives the VO its cone shape. The _closest_ velocity disk is
    # at D(-x/tau, r/tau), and this truncates the VO.

    x = -(agent.position - collider.position)
    v = agent.velocity - collider.velocity
    r = agent.radius + collider.radius

    x_len_sq = norm_sq(x)

    if x_len_sq >= r * r:
        # We need to decide whether to project onto the disk truncating the VO
        # or onto the sides.
        #
        # The center of the truncating disk doesn't mark the line between
        # projecting onto the sides or the disk, since the sides are not
        # parallel to the displacement. We need to bring it a bit closer. How
        # much closer can be worked out by similar triangles. It works out
        # that the new point is at x/t cos(theta)^2, where theta is the angle
        # of the aperture (so sin^2(theta) = (r/||x||)^2).
        adjusted_center = x/t * (1 - (r*r)/x_len_sq)

        if dot(v - adjusted_center, adjusted_center) < 0:
            # v lies in the front part of the cone
            # print("front")
            # print("front", adjusted_center, x_len_sq, r, x, t)
            w = v - x/t
            u = normalized(w) * r/t - w
            n = normalized(w)
        else: # v lies in the rest of the cone
            # print("sides")
            # Rotate x in the direction of v, to make it a side of the cone.
            # Then project v onto that, and calculate the difference.
            leg_len = sqrt(x_len_sq - r*r)
            # The sign of the sine determines which side to project on.
            sine = copysign(r, det((v, x)))
            rot = array(((leg_len, sine),(-sine, leg_len)))
            rotated_x = rot.dot(x) / x_len_sq
            n = perp(rotated_x)
            if sine < 0:
                # Need to flip the direction of the line to make the
                # half-plane point out of the cone.
                n = -n
            # print("rotated_x=%s" % rotated_x)
            u = rotated_x * dot(v, rotated_x) - v;
            # print("u=%s" % u)
    else:
        # We're already intersecting. Pick the closest velocity to our
        # velocity that will get us out of the collision within the next
        # timestep.
        # print("intersecting")
        w = v - x/dt
        u = normalized(w) * r/dt - w
        n = normalized(w);
    return u, n;

class InfeasibleError(RuntimeError):
    """Raised if an LP problem has no solution."""
    pass


class Line(object):
    """A line in space."""
    def __init__(self, point, direction):
        super(Line, self).__init__()
        self.point = array(point)
        self.direction = normalized(array(direction))

    def __repr__(self):
        return "Line(%s, %s)" % (self.point, self.direction)


def halfplane_optimize(lines, optimal_point):
    """Find the point closest to optimal_point in the intersection of the
    closed half-planes defined by lines which are in Hessian normal form
    (point-normal form)."""
    # We implement the quadratic time (though linear expected given randomly
    # permuted input) incremental half-plane intersection algorithm as laid
    # out in http://www.mpi-inf.mpg.de/~kavitha/lecture3.ps
    point = optimal_point
    for i, line in enumerate(lines):
        # If this half-plane already contains the current point, all is well.
        if dot(point - line.point, line.direction) >= 0:
            # assert False, point
            continue

        # Otherwise, the new optimum must lie on the newly added line. Compute
        # the feasible interval of the intersection of all the lines added so
        # far with the current one.
        prev_lines = itertools.islice(lines, i)
        left_dist, right_dist = line_halfplane_intersect(line, prev_lines)

        # Now project the optimal point onto the line segment defined by the
        # the above bounds. This gives us our new best point.
        point = point_line_project(line, optimal_point, left_dist, right_dist)
    return point

def point_line_project(line, point, left_bound, right_bound):
    """Project point onto the line segment defined by line, which is in
    point-normal form, and the left and right bounds with respect to line's
    anchor point."""
    # print("left_bound=%s, right_bound=%s" % (left_bound, right_bound))
    new_dir = perp(line.direction)
    # print("new_dir=%s" % new_dir)
    proj_len = dot(point - line.point, new_dir)
    # print("proj_len=%s" % proj_len)
    clamped_len = numpy.clip(proj_len, left_bound, right_bound)
    # print("clamped_len=%s" % clamped_len)
    return line.point + new_dir * clamped_len

def line_halfplane_intersect(line, other_lines):
    """Compute the signed offsets of the interval on the edge of the
    half-plane defined by line that is included in the half-planes defined by
    other_lines.

    The offsets are relative to line's anchor point, in units of line's
    direction.

    """
    # We use the line intersection algorithm presented in
    # http://stackoverflow.com/a/565282/126977 to determine the intersection
    # point. "Left" is the negative of the canonical direction of the line.
    # "Right" is positive.
    left_dist = float("-inf")
    right_dist = float("inf")
    for prev_line in other_lines:
        num1 = dot(prev_line.direction, line.point - prev_line.point)
        den1 = det((line.direction, prev_line.direction))
        # num2 = det((perp(prev_line.direction), line.point - prev_line.point))
        # den2 = det((perp(line.direction), perp(prev_line.direction)))

        # assert abs(den1 - den2) < 1e-6, (den1, den2)
        # assert abs(num1 - num2) < 1e-6, (num1, num2)

        num = num1
        den = den1

        # Check for zero denominator, since ZeroDivisionError (or rather
        # FloatingPointError) won't necessarily be raised if using numpy.
        if den == 0:
            # The half-planes are parallel.
            if num < 0:
                # The intersection of the half-planes is empty; there is no
                # solution.
                raise InfeasibleError
            else:
                # The *half-planes* intersect, but their lines don't cross, so
                # ignore.
                continue

        # Signed offset of the point of intersection, relative to the line's
        # anchor point, in units of the line's direction.
        offset = num / den
        if den > 0:
            # Point of intersection is to the right.
            right_dist = min((right_dist, offset))
        else:
            # Point of intersection is to the left.
            left_dist = max((left_dist, offset))

        if left_dist > right_dist:
            # The interval is inconsistent, so the feasible region is empty.
            raise InfeasibleError
    return left_dist, right_dist

agents = []
for i in range(N_AGENTS):
    theta = 2 * pi * i / N_AGENTS
    x = RADIUS * array((cos(theta), sin(theta))) #+ random.uniform(-1, 1)
    vel = normalized(-x) * SPEED
    pos = (random.uniform(-20, 20), random.uniform(-20, 20))
    agents.append(Agent(pos, (0., 0.), 1., SPEED, vel))


colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
]

pygame.init()

dim = (640, 480)
screen = pygame.display.set_mode(dim)

O = array(dim) / 2  # Screen position of origin.
scale = 6  # Drawing scale.

clock = pygame.time.Clock()
FPS = 20
dt = 1/FPS
tau = 5

def draw_agent(agent, color):
    pygame.draw.circle(screen, color, rint(agent.position * scale + O).astype(int), int(round(agent.radius * scale)), 0)

def draw_orca_circles(a, b):
    for x in linspace(0, tau, 21):
        if x == 0:
            continue
        pygame.draw.circle(screen, pygame.Color(0, 0, 255), rint((-(a.position - b.position) / x + a.position) * scale + O).astype(int), int(round((a.radius + b.radius) * scale / x)), 1)

def draw_velocity(a):
    pygame.draw.line(screen, pygame.Color(0, 255, 255), rint(a.position * scale + O).astype(int), rint((a.position + a.velocity) * scale + O).astype(int), 1)
    # pygame.draw.line(screen, pygame.Color(255, 0, 255), rint(a.position * scale + O).astype(int), rint((a.position + a.pref_velocity) * scale + O).astype(int), 1)

running = True
accum = 0
all_lines = [[]] * len(agents)
while running:
    accum += clock.tick(FPS)

    while accum >= dt * 1000:
        accum -= dt * 1000

        new_vels = [None] * len(agents)
        for i, agent in enumerate(agents):
            candidates = agents[:i] + agents[i + 1:]
            # print(candidates)
            new_vels[i], all_lines[i] = orca(agent, candidates, tau, dt)
            # print(i, agent.velocity)

        for i, agent in enumerate(agents):
            agent.velocity = new_vels[i]
            agent.position += agent.velocity * dt

    screen.fill(pygame.Color(0, 0, 0))

    for agent in agents[1:]:
        draw_orca_circles(agents[0], agent)

    for agent, color in zip(agents, itertools.cycle(colors)):
        draw_agent(agent, color)
        draw_velocity(agent)
        # print(sqrt(norm_sq(agent.velocity)))

    for line in all_lines[0]:
        # Draw ORCA line
        alpha = agents[0].position + line.point + perp(line.direction) * 100
        beta = agents[0].position + line.point + perp(line.direction) * -100
        pygame.draw.line(screen, (255, 255, 255), rint(alpha * scale + O).astype(int), rint(beta * scale + O).astype(int), 1)

        # Draw normal to ORCA line
        gamma = agents[0].position + line.point
        delta = agents[0].position + line.point + line.direction
        pygame.draw.line(screen, (255, 255, 255), rint(gamma * scale + O).astype(int), rint(delta * scale + O).astype(int), 1)

    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
pygame.quit()

