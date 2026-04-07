# Integral Lookup Tables

Place fog integral lookup table files (`*.pickle`) under:

- `integral_lookup_tables/original/`

Expected filename pattern:

- `integral_0m_to_200m_stepsize_0.1m_tau_h_20ns_alpha_<alpha>.pickle`

`fog_simulation.py` will automatically load the nearest available `alpha` file for each frame.
If no table is found, soft-fog replacement is skipped and only hard attenuation is applied.
