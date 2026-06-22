"""Shared UI layer for the ML4B Streamlit app — the "Night Scope" design system.

This package holds ONLY presentation code (CSS, reusable HTML/SVG components and
Plotly styling). It contains no machine-learning logic: every prediction still
comes from ``src/ml4b/`` so the training and app pipelines stay identical
(see CLAUDE.md, the single most important architectural rule).

Modules:
  theme  — colour/typography tokens, global CSS injection and HTML components
  viz    — dark-themed Plotly figures (oscilloscope, timelines, charts)
"""
