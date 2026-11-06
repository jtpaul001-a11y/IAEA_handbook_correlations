import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})


def neutron_yield(E_d, theta, Y0, Yi):
    """
    Calculate neutron yield:

        Y_n(E_d, theta) =
        Y0(E_d) + sum_i [Yi(E_d) * cos(theta)^i]

    Parameters
    ----------
    E_d : float
        Incident deuteron energy.
    theta : float
        Neutron emission angle in radians.
    Y0 : float
        Y0(E_d), the zeroth-order yield coefficient.
    Yi : array-like
        Coefficients [Y1(E_d), Y2(E_d), ..., Yn(E_d)].

    Returns
    -------
    Y_n : float
        Neutron yield.
    """

    cos_theta = np.cos(theta)

    Y_n = Y0

    for i, Y_i in enumerate(Yi, start=1):
        Y_n += Y_i * cos_theta**i

    return Y_n


def neutron_energy(E_d, theta, E0, Ei):
    """
    Calculate neutron energy:

        E_n(E_d, theta) =
        E0(E_d) + sum_i [Ei(E_d) * cos(theta)^i]

    Parameters
    ----------
    E_d : float
        Incident deuteron energy.
    theta : float
        Neutron emission angle in radians.
    E0 : float
        E0(E_d), the zeroth-order energy coefficient.
    Ei : array-like
        Coefficients [E1(E_d), E2(E_d), ..., En(E_d)].

    Returns
    -------
    E_n : float
        Neutron energy.
    """

    cos_theta = np.cos(theta)

    E_n = E0

    for i, E_i in enumerate(Ei, start=1):
        E_n += E_i * cos_theta**i

    return E_n




## These are the thick target coefficients for DT reaction, taken from iaea handbook. ##
energy_keV = np.array([50, 100, 150, 200, 250, 300, 325], dtype=float)

A0 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)
A1 = np.array([0.03003, 0.04087, 0.04727, 0.05124, 0.05419, 0.05651, 0.05616], dtype=float)
A2 = np.array([0.00035, 0.00062, 0.00083, 0.00096, 0.00110, 0.00119, 0.00119], dtype=float)

yield_coefficients_D_T = {
    "energy_keV": energy_keV,
    "A0": A0,
    "A1": A1,
    "A2": A2
}


E_d_keV_thin = np.array([20, 30, 40, 50, 60, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500], dtype=float)
A0_thin = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)
A1_thin = np.array([0.0220, 0.0227, 0.0310, 0.0344, 0.0518, 0.0407, 0.0482, 0.0599, 0.0678, 0.0685, 0.0818, 0.0904, 0.1003, 0.1140, 0.1273], dtype=float)
A2_thin = np.array([0.00025, -0.0093, 0.0007, 0.0010, -0.0035, 0.0011, 0.0011, 0.0009, 0.0005, -0.0104, 0.0005, 0.0028, -0.0008, -0.0101, -0.0187], dtype=float)

yield_coefficients_D_T_thin = {
    "energy_keV": E_d_keV_thin,
    "A0": A0_thin,
    "A1": A1_thin,
    "A2": A2_thin,
}


E_d_keV = np.array([50, 100, 150, 200, 250, 300, 325], dtype=float)
E0 = np.array([14.06520, 14.07883, 14.08942, 14.09680, 14.10286, 14.10803, 14.10723], dtype=float)
E1 = np.array([0.42329, 0.57613, 0.66776, 0.72427, 0.76661, 0.80001, 0.79477], dtype=float)
E2 = np.array([0.00682, 0.01222, 0.01600, 0.01908, 0.02167, 0.02374, 0.02347], dtype=float)

energy_coefficients_D_T = {
    "energy_keV": E_d_keV,
    "E0": E0,
    "E1": E1,
    "E2": E2
}


E_d_keV_thin = np.array([50, 100, 200, 300, 400, 500], dtype=float)
E0_thin = np.array([14.04814, 14.06732, 14.10711, 14.14704, 14.18670, 14.22569], dtype=float)
E1_thin = np.array([0.47679, 0.67488, 0.95596, 1.17282, 1.35640, 1.51899], dtype=float)
E2_thin = np.array([0.00834, 0.01719, 0.03320, 0.04923, 0.06527, 0.08249], dtype=float)

energy_coefficients_D_T_thin = {
    "energy_keV": E_d_keV_thin,
    "E0": E0_thin,
    "E1": E1_thin,
    "E2": E2_thin,
}


def _interpolate_coefficients(E_d, coeffs):
    """Interpolate a coefficient set at a single deuteron energy."""
    energy_grid = coeffs["energy_keV"]
    if E_d < energy_grid.min() or E_d > energy_grid.max():
        raise ValueError(
            f"Deuteron energy {E_d} keV is outside the tabulated range "
            f"{energy_grid.min()}–{energy_grid.max()} keV."
        )

    return {
        name: np.interp(E_d, energy_grid, values)
        for name, values in coeffs.items()
        if name != "energy_keV"
    }


def plot_neutron_yield_and_energy(E_d_keV, angle_min_deg=0.0, angle_max_deg=170.0,
                                 n_points=181, show=True, thin_target=False):
    """
    Plot neutron yield and neutron energy versus emission angle for a given
    deuteron energy using the tabulated DT correlation coefficients.

    Parameters
    ----------
    E_d_keV : float
        Incident deuteron energy in keV.
    angle_min_deg : float, optional
        Minimum emission angle in degrees, default 0.
    angle_max_deg : float, optional
        Maximum emission angle in degrees, default 170.
    n_points : int, optional
        Number of angle points to evaluate, default 181.
    show : bool, optional
        If True, display the plot immediately.
    thin_target : bool, optional
        If True, use the thin-target DT coefficient dictionaries.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax_yield, ax_energy : matplotlib.axes.Axes
        The two plot axes.
    """
    angles_deg = np.linspace(angle_min_deg, angle_max_deg, n_points)
    theta = np.deg2rad(angles_deg)

    yield_coefficients = yield_coefficients_D_T_thin if thin_target else yield_coefficients_D_T
    energy_coefficients = energy_coefficients_D_T_thin if thin_target else energy_coefficients_D_T

    yield_coeffs = _interpolate_coefficients(E_d_keV, yield_coefficients)
    energy_coeffs = _interpolate_coefficients(E_d_keV, energy_coefficients)

    Y0 = yield_coeffs["A0"]
    Yi = [yield_coeffs["A1"], yield_coeffs["A2"]]
    E0 = energy_coeffs["E0"]
    Ei = [energy_coeffs["E1"], energy_coeffs["E2"]]

    yield_vals = neutron_yield(E_d_keV, theta, Y0, Yi)
    energy_vals = neutron_energy(E_d_keV, theta, E0, Ei)

    fig, (ax_yield, ax_energy) = plt.subplots(1, 2, figsize=(12, 4))

    ax_yield.plot(angles_deg, yield_vals, color='tab:blue', linewidth=2)
    ax_yield.set_xlabel('Neutron emission angle [deg]')
    ax_yield.set_ylabel('Neutron yield')
    target_label = 'thin-target' if thin_target else 'thick-target'
    ax_yield.set_title(f'D-T {target_label} yield at E_d = {E_d_keV} keV')
    ax_yield.grid(True, alpha=0.3)

    ax_energy.plot(angles_deg, energy_vals, color='tab:orange', linewidth=2)
    ax_energy.set_xlabel('Neutron emission angle [deg]')
    ax_energy.set_ylabel('Neutron energy [MeV]')
    ax_energy.set_title(f'D-T {target_label} neutron energy at E_d = {E_d_keV} keV')
    ax_energy.grid(True, alpha=0.3)

    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax_yield, ax_energy


def plot_neutron_yield_and_energy_by_incident_energy(E_d_keV_values, angle_min_deg=0.0,
                                                     angle_max_deg=170.0, n_points=181,
                                                     cmap_name='viridis', show=True,
                                                     thin_target=False):
    """
    Plot neutron yield and neutron energy for multiple deuteron energies,
    coloring each curve by incident energy and adding a shared colorbar.

    Parameters
    ----------
    E_d_keV_values : array-like
        Incident deuteron energies in keV.
    angle_min_deg : float, optional
        Minimum emission angle in degrees, default 0.
    angle_max_deg : float, optional
        Maximum emission angle in degrees, default 170.
    n_points : int, optional
        Number of angle points to evaluate, default 181.
    cmap_name : str, optional
        Matplotlib colormap name, default 'viridis'.
    show : bool, optional
        If True, display the plot immediately.
    thin_target : bool, optional
        If True, use the thin-target DT coefficient dictionaries.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax_yield, ax_energy : matplotlib.axes.Axes
        The two plot axes.
    """
    E_d_keV_values = np.asarray(E_d_keV_values, dtype=float)
    angles_deg = np.linspace(angle_min_deg, angle_max_deg, n_points)
    theta = np.deg2rad(angles_deg)

    yield_coefficients = yield_coefficients_D_T_thin if thin_target else yield_coefficients_D_T
    energy_coefficients = energy_coefficients_D_T_thin if thin_target else energy_coefficients_D_T

    fig, (ax_yield, ax_energy) = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True,
                                               constrained_layout=True)
    cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(E_d_keV_values.min(), E_d_keV_values.max())

    for E_d_keV in E_d_keV_values:
        yield_coeffs = _interpolate_coefficients(E_d_keV, yield_coefficients)
        energy_coeffs = _interpolate_coefficients(E_d_keV, energy_coefficients)

        Y0 = yield_coeffs["A0"]
        Yi = [yield_coeffs["A1"], yield_coeffs["A2"]]
        E0 = energy_coeffs["E0"]
        Ei = [energy_coeffs["E1"], energy_coeffs["E2"]]

        yield_vals = neutron_yield(E_d_keV, theta, Y0, Yi)
        energy_vals = neutron_energy(E_d_keV, theta, E0, Ei)
        color = cmap(norm(E_d_keV))

        ax_yield.plot(angles_deg, yield_vals, color=color, linewidth=2)
        ax_energy.plot(angles_deg, energy_vals, color=color, linewidth=2, label=f'{E_d_keV:g} keV')

    ax_yield.set_xlabel('Neutron emission angle [deg]')
    ax_yield.set_ylabel('Neutron yield')
    target_label = 'thin-target' if thin_target else 'thick-target'
    ax_yield.set_title(f'D-T {target_label} neutron yield vs angle')
    ax_yield.grid(True, alpha=0.3)

    ax_energy.set_xlabel('Neutron emission angle [deg]')
    ax_energy.set_ylabel('Neutron energy [MeV]')
    ax_energy.set_title(f'D-T {target_label} neutron energy vs angle')
    ax_energy.grid(True, alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_yield, ax_energy], pad=0.02)
    cbar.set_label('Incident deuteron energy [keV]')

    if show:
        plt.show()

    return fig, ax_yield, ax_energy


def plot_thick_and_thin_neutron_yield_and_energy(E_d_keV_values, angle_min_deg=0.0,
                                                 angle_max_deg=170.0, n_points=181,
                                                 show=True, cmap_name='viridis'):
    """
    Plot thick-target and thin-target neutron yield and energy versus angle
    for one or more incident deuteron energies.

    Parameters
    ----------
    E_d_keV_values : float or array-like
        One or more incident deuteron energies in keV.
    angle_min_deg : float, optional
        Minimum emission angle in degrees, default 0.
    angle_max_deg : float, optional
        Maximum emission angle in degrees, default 170.
    n_points : int, optional
        Number of angle points to evaluate, default 181.
    show : bool, optional
        If True, display the plot immediately.
    cmap_name : str, optional
        Matplotlib colormap name used to color the incident energies.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax_yield, ax_energy : matplotlib.axes.Axes
        The two plot axes.
    """
    E_d_keV_values = np.atleast_1d(np.asarray(E_d_keV_values, dtype=float))
    angles_deg = np.linspace(angle_min_deg, angle_max_deg, n_points)
    theta = np.deg2rad(angles_deg)

    fig, (ax_yield, ax_energy) = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True,
                                               constrained_layout=True)
    cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(E_d_keV_values.min(), E_d_keV_values.max())

    target_specs = [
        ("Thick target", yield_coefficients_D_T, energy_coefficients_D_T, '-'),
        ("Thin target", yield_coefficients_D_T_thin, energy_coefficients_D_T_thin, '--'),
    ]

    for E_d_keV in E_d_keV_values:
        color = cmap(norm(E_d_keV))

        for target_label, yield_coefficients, energy_coefficients, linestyle in target_specs:
            yield_coeffs = _interpolate_coefficients(E_d_keV, yield_coefficients)
            energy_coeffs = _interpolate_coefficients(E_d_keV, energy_coefficients)

            Y0 = yield_coeffs["A0"]
            Yi = [yield_coeffs["A1"], yield_coeffs["A2"]]
            E0 = energy_coeffs["E0"]
            Ei = [energy_coeffs["E1"], energy_coeffs["E2"]]

            yield_vals = neutron_yield(E_d_keV, theta, Y0, Yi)
            energy_vals = neutron_energy(E_d_keV, theta, E0, Ei)

            label = f'{E_d_keV:g} keV {target_label}'
            ax_yield.plot(angles_deg, yield_vals, color=color, linestyle=linestyle,
                          linewidth=2, label=label)
            ax_energy.plot(angles_deg, energy_vals, color=color, linestyle=linestyle,
                           linewidth=2, label=label)

    ax_yield.set_xlabel('Neutron emission angle [deg]')
    ax_yield.set_ylabel('Neutron yield')
    ax_yield.set_title('D-T neutron yield vs angle')
    ax_yield.grid(True, alpha=0.3)
    ax_yield.legend()

    ax_energy.set_xlabel('Neutron emission angle [deg]')
    ax_energy.set_ylabel('Neutron energy [MeV]')
    ax_energy.set_title('D-T neutron energy vs angle')
    ax_energy.grid(True, alpha=0.3)
    ax_energy.legend()

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_yield, ax_energy], pad=0.02)
    cbar.set_label('Incident deuteron energy [keV]')

    if show:
        plt.show()

    return fig, ax_yield, ax_energy




def plot_experimental_vs_handbook_energy_vs_incident_energy(
    csv_path,
    incident_energy_min_keV=100.0,
    incident_energy_max_keV=150.0,
    angle_deg=None,
    show=True,
    thick_target=True,
):
    """
    Compare experimental peak energies against the DT handbook neutron-energy model.

    This is only physically meaningful on the x-axis of incident deuteron energy when
    the CSV contains an 'E_d_keV' column. If the CSV does not have that column, the
    function raises a clear error rather than plotting a misleading graph.
    """
    experimental_df = pd.read_csv(csv_path)

    required_cols = {'Angle', 'E_peak'}
    missing = required_cols.difference(experimental_df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {sorted(missing)}. "
            "Expected at least 'Angle' and 'E_peak'."
        )

    if 'E_d_keV' not in experimental_df.columns:
        raise ValueError(
            "The CSV does not contain an 'E_d_keV' column, so it cannot be plotted "
            "as 'experimental energy vs incident energy'. Add that column or plot "
            "'experimental energy vs angle' instead."
        )

    x_exp = experimental_df['E_d_keV'].to_numpy(dtype=float)
    y_exp = experimental_df['E_peak'].to_numpy(dtype=float)
    x_label = 'Incident deuteron energy [keV]'

    if angle_deg is None:
        angle_deg = np.unique(np.round(experimental_df['Angle'].to_numpy(dtype=float), 1))
    angle_deg = np.atleast_1d(np.asarray(angle_deg, dtype=float))

    coeffs_to_use = energy_coefficients_D_T if thick_target else energy_coefficients_D_T_thin
    E_d_vals = np.linspace(incident_energy_min_keV, incident_energy_max_keV, 300)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(x_exp, y_exp, s=60, color='tab:red', alpha=0.85,
               label='Experimental peak energy')

    for angle in angle_deg:
        theta = np.deg2rad(angle)
        handbook_energy = []
        for E_d_keV in E_d_vals:
            coeffs = _interpolate_coefficients(E_d_keV, coeffs_to_use)
            E_n = neutron_energy(E_d_keV, theta, coeffs['E0'], [coeffs['E1'], coeffs['E2']])
            handbook_energy.append(E_n)
        ax.plot(E_d_vals, handbook_energy, linewidth=2, label=f'Handbook @ {angle:.1f} deg')

    ax.set_xlabel(x_label)
    ax.set_ylabel('Neutron energy [MeV]')
    target_label = 'thick-target' if thick_target else 'thin-target'
    ax.set_title(
        f'Experimental peak energy vs DT handbook model ({target_label}, '
        f'{incident_energy_min_keV:g}-{incident_energy_max_keV:g} keV)'
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_handbook_energy_vs_angle_overlay(csv_path, E_d_values=None,
                                         angle_min_deg=None, angle_max_deg=None,
                                         show=True, thick_target=True):
    """
    Plot the DT handbook neutron energy vs emission angle for a set of incident
    deuteron energies, then overlay the experimental CSV data measured for about
    120 keV.
    """
    experimental_df = pd.read_csv(csv_path)

    required_cols = {'Angle', 'E_peak'}
    missing = required_cols.difference(experimental_df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {sorted(missing)}. "
            "Expected at least 'Angle' and 'E_peak'."
        )

    if E_d_values is None:
        E_d_values = np.arange(100, 151, 10, dtype=float)
    E_d_values = np.atleast_1d(np.asarray(E_d_values, dtype=float))

    angle_values = experimental_df['Angle'].to_numpy(dtype=float)
    if angle_min_deg is None:
        angle_min_deg = float(np.min(angle_values))
    if angle_max_deg is None:
        angle_max_deg = float(np.max(angle_values))

    angle_grid = np.linspace(angle_min_deg, angle_max_deg, 400)
    coeffs_dict = energy_coefficients_D_T if thick_target else energy_coefficients_D_T_thin

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(E_d_values.min(), E_d_values.max())

    for E_d_keV in E_d_values:
        handbook_energy = []
        for angle_deg in angle_grid:
            coeffs = _interpolate_coefficients(E_d_keV, coeffs_dict)
            theta = np.deg2rad(angle_deg)
            E_n = neutron_energy(E_d_keV, theta, coeffs['E0'], [coeffs['E1'], coeffs['E2']])
            handbook_energy.append(E_n)
        ax.plot(angle_grid, handbook_energy, linewidth=2, color=cmap(norm(E_d_keV)),
                label=f'{E_d_keV:g} keV')

    ax.scatter(angle_values, experimental_df['E_peak'].to_numpy(dtype=float),
               color='tab:red', s=50, label='Experimental data', zorder=3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.03)
    cbar.set_label('Incident deuteron energy [keV]')

    ax.set_xlabel('Angle [deg]')
    ax.set_ylabel('Neutron energy [MeV]')
    target_label = 'thick-target' if thick_target else 'thin-target'
    ax.set_title(f'Iaea handbook neutron energy emission vs experimental data({target_label})')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax




plot_handbook_energy_vs_angle_overlay(
    '/home/jonathan-paul/260414_Starfire_nGen_characterization/analysis/E_peak_and_FWHM_vs_angle.csv',
    E_d_values=np.arange(50, 121, 10),
    show=True,
    thick_target=True,
)

