import numpy as np
import scipy.signal

def generate_uniform_parametric_eq(num_points, max_order, rng):
    """Generate a random parametric EQ cascase according to the method specified in
    [Nercessian 2020](https://dafx2020.mdw.ac.at/proceedings/papers/DAFx2020_paper_7.pdf).

    Returns:
        coef
        mag
        phs
    """
    zeros = []
    poles = []
    sos_holder = []
    -10
    num_peaks = (max_order) // 2 - 2  # Number of peaking filters to use paper=10

    omega_low = rng.uniform(low=0.0, high=np.pi)
    g = rng.uniform(low=-10.0, high=10.0)
    q = rng.uniform(low=0.1, high=1.0)
    A = np.power(10, g / 40)
    alpha = np.sin(omega_low) * np.sqrt((A ** 2 + 1) * ((1 / q) - 1) + 2 * A)

    b0 = A * ((A + 1) - (A - 1) * np.cos(omega_low) + alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(omega_low))
    b2 = A * ((A + 1) - (A - 1) * np.cos(omega_low) - alpha)

    a0 = (A + 1) + (A - 1) * np.cos(omega_low) + alpha
    a1 = -2 * A * ((A - 1) + (A + 1) * np.cos(omega_low))
    a2 = (A + 1) + (A - 1) * np.cos(omega_low) - alpha

    sos_poly = np.asarray([b0, b1, b2, a0, a1, a2])
    sos_holder.append(sos_poly)
    num_poly = np.asarray([b0, b1, b2])
    zeros.append(num_poly)
    den_poly = np.asarray([a0, a1, a2])
    poles.append(den_poly)

    omega_high = rng.uniform(low=0.0, high=np.pi)
    g = rng.uniform(low=-10.0, high=10.0)
    q = rng.uniform(low=0.1, high=1.0)
    A = np.power(10, g / 40)
    alpha = np.sin(omega_high) * np.sqrt((A ** 2 + 1) * ((1 / q) - 1) + 2 * A)

    b0 = A * ((A + 1) + (A - 1) * np.cos(omega_high) + alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(omega_high))
    b2 = A * ((A + 1) + (A - 1) * np.cos(omega_high) - alpha)

    a0 = (A + 1) - (A - 1) * np.cos(omega_high) + alpha
    a1 = 2 * A * ((A - 1) - (A + 1) * np.cos(omega_high))
    a2 = (A + 1) - (A - 1) * np.cos(omega_high) - alpha

    sos_poly = np.asarray([b0, b1, b2, a0, a1, a2])
    sos_holder.append(sos_poly)
    num_poly = np.asarray([b0, b1, b2])
    zeros.append(num_poly)
    den_poly = np.asarray([a0, a1, a2])
    poles.append(den_poly)

    ##Peaking Filters
    for jj in range(num_peaks):
        omega = rng.uniform(low=0.0, high=np.pi)
        g = rng.uniform(low=-10, high=10)
        q = rng.uniform(low=0.1, high=3.0)

        alpha = np.sin(omega) / (2 * q)
        A = np.power(10, g / 40)

        b0 = 1 + (alpha * A)
        b1 = -2 * np.cos(omega)
        b2 = 1 - (alpha * A)

        a0 = 1 + (alpha / A)
        a1 = -2 * np.cos(omega)
        a2 = 1 - (alpha / A)

        sos_poly = np.asarray([b0, b1, b2, a0, a1, a2])
        sos_holder.append(sos_poly)
        num_poly = np.asarray([b0, b1, b2])
        zeros.append(num_poly)
        den_poly = np.asarray([a0, a1, a2])
        poles.append(den_poly)

    sos = np.vstack(sos_holder)
    my_norms = sos[:, 3]
    sos = sos / my_norms[:, None]  ##sosfreqz requires sos[:,3]=1
    w, h = scipy.signal.sosfreqz(sos, worN=num_points)
    return h

