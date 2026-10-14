# Metropolis-Hastings

Run MH alone with `python3 dmri_project_mh.py --force-recompute`.
Its defaults are 10,000 samples, seed 0, gamma=0.01, nu=2000,
and a mean-centered Wishart proposal. It discards 1,000 burn-in iterations
before computing posterior summaries and saving its posterior and trace figures.

The MH stage in `dmri_project_complete.py` uses the same settings and reports
acceptance rate, credible intervals, ESS, and fresh MH inference runtime.
The other methods retain their upstream execution settings.

Run MH regression tests with
`python3 -m unittest test_dmri_project_mh test_dmri_project_complete`.
