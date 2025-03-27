# gθ, is a graph
# (V, E), which represents the physical system
# Nv bodies are represented as nodes
# V = {vi}i=1:Nv
# E = {(sk,rk, ek)}k=1:N, relationships between pairs of bodies are represented as directed edges
# vi node attribute contains a trainable scalar variable that is fixed across all input graphs and is analogous to mass
# sk are sender nodes
# rk are receiver nodes
# ek edge attribute is the spatial displacement vector between the two corresponding bodies

import math
import random
# use argparse to get input values
# walkthough of orbital mechanics git hub


G = 6.67430 * (10**-11)  # This is gravitational constant
"""
r is radias from center of object
M is mass of the central body
m is mass of the orbiting object
a is semi-major axis of orbit
b is semi-minor axis of orbit
n is gravitational influence of planatary body
i is gravitational influence of planatary body
"""


def main_variables(self, variable_id, value=None) -> float:
    if value is None:
        value = random.uniform

    self.set_variable(variable_id, "r", value(0, 10000))
    self.set_variable(variable_id, "M", value(0, 10000))
    self.set_variable(variable_id, "m", value(0, 10000))
    self.set_variable(variable_id, "a", value(0, 384400))
    self.set_variable(variable_id, "b", value(0, 383800))
    self.set_variable(variable_id, "n", value(0, 10000))
    self.set_variable(variable_id, "i", value(0, 10000))


# Newtons second law
# Force is mass * times acceleration
def newtons_second_law(mass: int, acceleration: float) -> float:
    """
    mass represents the mass of the body

    acceleration represents the speed of which the body is accelerating
    """
    return mass * acceleration


# newtons third law
def newtons_third_law(force_A_on_B: float) -> float:
    """
    Force B on A, is negative A on B
    """
    return -(force_A_on_B)


# Newton's Law of Universal Gravitation
# F is the gravitational force between the two objects
# G is gravitation constant, 6.67430×((10)^−11)


def newtons_uni_grav_law(m1: float, m2: float, rd: float, G: float) -> float:
    """
    m1 and m2 are masses of the two objects respectivly

    rd is the distance between the centers of the two objects
    """
    return (G * m1 * m2) / (rd**2)


# Orbital Velocity
# G is gravitation constant, 6.67430×((10)^−11)


def Orbital_velocity(G: float, M: float, r: float) -> float:
    """
    M is the mass of the central body

    r is the radius of the orbit(from center of central body)
    """
    return math.sqrt((G * M) / r)


# Orbital Energy
# E Total mechanical energy of orbiting object
# U Potential energy
# K kinetic energy
# G is gravitation constant, 6.67430×((10)^−11)
# M is the mass of the central body
# r is the distance from the central body
# m is the mass of the orbiting object


def Orbital_energy(G: float, M: float, m: float, r: float) -> float:
    """
    this represents the orbital energy, which two plants would have on each other
    """
    return -((G * M * m) / (2 * r))


# Vis-Viva Equation
# provides the orbital speed at any point in an orbit
# vr is orbital velocity at distance r
# G is gravitation constant, 6.67430×((10)^−11)
# M is the mass of the central body
# r is the radius of the orbit(from center of central body)
# a is the semi-major axis of the orbit


def Vis_viva(G: float, M: float, r: float, a: float) -> float:
    return math.sqrt((G * M) * ((2 / r) - (1 / a)))


# Kepler's First Law (Law of Ellipses)
# rθ is the distance from the Sun to the planet at a given angle θ
# a is the semi-major axis (the longest radius of the ellipse)


def Kep_first_law(a: float, e: float, θ: float) -> float:
    """
    e is the orbital eccentricity (which describes the "flattening" of the ellipse)
    θ is the true anomaly (the angle of the planet relative to the closest point of the orbit, also known as perihelion)
    """

    return (a * (1 - (e**2))) / (1 + (e * (math.cos(θ))))


# Escape Velocity
# ve is the escape velocity
# G is gravitation constant, 6.67430×((10)^−11)
# M is the mass of the central body
# r is the radius of the orbit(from center of central body)


def Escape_vel(G: float, M: float, r: float) -> float:
    return math.sqrt((2 * G * M) / (r))


# grav. influence
# GI represents gravitational influence as the sum of gravitational potentials experienced by all other bodies
# G is gravitation constant, 6.67430×((10)^−11)
# n represents teh planitary bodies
# Mn is the masses of objects n respectivly
# ri is the position vector of object i in the graph
# rn is the position vector of object n in the graph
# gravitational potential is negative which is why theirs a negative sign
# ∑(i != n) represents the sum potential over all bodies in the system, except for the body n itself
# ∑_GI is the sum of all gravitational influence minus body n
# 1 would represent planet n where as n+1 would represent planet i


def grav_infl(Mn: float, rn: float, ri: float, i: float, n: float) -> float:
    GI = (-G * Mn) / (abs(rn - ri))
    for Gi in range(n):
        if i != n:
            GI = sum(range(1, n + 1))
        else:
            GI = newtons_uni_grav_law
    return GI
