"""
Early Kalman filter sketch placeholders.

State vectors (to refine later):
- Window node: x_w = [T_window, H_window]
- Bed node: x_b = [T_bed, H_bed]
"""


def simulate_window_dynamics():
    """
    TODO: simulate T_window evolution:
    T_next = T + a*(T_out - T)*window_open_frac + ß*fan_on
    """
    pass


def simulate_bedside_dynamics():
    """
    TODO: simulate H_bed evolution:
    H_next = H + ?*humidifier_power + d*(H_window - H_bed)
    """
    pass


def main() -> None:
    print("Kalman sketch placeholder")


if __name__ == "__main__":
    main()