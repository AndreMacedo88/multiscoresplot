# Color Space

Color mapping from gene set scores to RGB (pipeline steps 2–3).

Both `blend_to_rgb` and `reduce_to_rgb` return an `RGBResult` object that wraps
the RGB array with metadata (method, gene set names, colors). This metadata is
auto-detected by the plotting functions.

## RGBResult

::: multiscoresplot.RGBResult

---

## Blending (2–3 gene sets)

::: multiscoresplot.blend_to_rgb

---

## Dimensionality Reduction (2+ gene sets)

::: multiscoresplot.reduce_to_rgb

---

## Custom Reducers

::: multiscoresplot.register_reducer

---

## Utility

::: multiscoresplot.get_component_labels
