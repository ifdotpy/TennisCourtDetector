import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

from scipy.optimize import minimize, differential_evolution


class Court:
    # scale is calculated to be 9. it is equal to 50mm
    def __init__(self, scale=9, gap_width=1):
        self.scale = scale
        self.gap_width = gap_width
        self.n, self.m = 52 * scale + gap_width, 24 * scale + gap_width
        self.a = np.zeros((self.n, self.m))

    def add_borders(self):
        gap = self.gap_width
        self.a[gap:self.n - 1 - gap, gap] = 1
        self.a[gap:self.n - 1 - gap, self.m - 1 - gap] = 1
        self.a[gap, gap:self.m - 1 - gap] = 1
        self.a[self.n - 1 - gap, gap:self.m - 1 - gap] = 1

    def add_singles(self):
        gap = self.gap_width
        dist = 3 * self.scale
        self.a[gap:self.n - gap, self.m - 1 - dist - gap] = 1
        self.a[gap:self.n - gap, dist + gap] = 1

    def add_service_lines(self):
        v_dist = 12 * self.scale
        h_dist = 3 * self.scale
        self.a[self.n - 1 - v_dist, h_dist:self.m - 1 - h_dist] = 1
        self.a[v_dist, h_dist:self.m - 1 - h_dist] = 1

    def add_center_line(self):
        v_dist = 12 * self.scale
        self.a[v_dist:self.n - 1 - v_dist, self.m // 2] = 1

    def get_intersections(self):
        gap = self.gap_width
        dist = 3 * self.scale
        v_dist = 12 * self.scale
        h_dist = 3 * self.scale
        intersections = [
            (gap, gap),  # Top-left corner
            (gap, self.m - 1 - gap),  # Top-right corner
            (self.n - 1 - gap, gap),  # Bottom-left corner
            (self.n - 1 - gap, self.m - 1 - gap),  # Bottom-right corner
            (gap, self.m - 1 - dist - gap),  # Top-left singles
            (self.n - 1 - gap, self.m - 1 - dist - gap),  # Bottom-left singles
            (gap, dist + gap),  # Top-right singles
            (self.n - 1 - gap, dist + gap),  # Bottom-right singles
            (self.n - 1 - v_dist, h_dist),  # Top-left service line
            (self.n - 1 - v_dist, self.m - 1 - h_dist),  # Top-right service line
            (v_dist, h_dist),  # Bottom-left service line
            (v_dist, self.m - 1 - h_dist),  # Bottom-right service line
            (v_dist, self.m // 2),  # Center line top
            (self.n - 1 - v_dist, self.m // 2)  # Center line bottom
        ]
        return intersections


def main():
    court = Court()
    # court.add_borders()
    # court.add_singles()
    # court.add_service_lines()
    # court.add_center_line()
    print(court.get_intersections())


if __name__ == "__main__":
    main()
