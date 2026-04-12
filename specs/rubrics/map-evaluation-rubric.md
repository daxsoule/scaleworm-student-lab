# Map Evaluation Rubric

A shared standard for evaluating cartographic figures across all projects. Derived from the MISO ASHES Vent Field detail map (the first map to pass all criteria).

## Sizing Tiers

Absolute font sizes depend on the output medium. Each project constitution must declare which tier applies.

| Tier | Use Case | Title | Axis/Caption | Feature Labels | Min DPI |
|------|----------|-------|--------------|----------------|---------|
| **Poster** | Conference posters, large-format prints | >= 24pt | >= 18pt | >= 11pt | 600 |
| **Paper** | Journal figures, reports | >= 14pt | >= 10pt | >= 8pt | 300 |
| **Presentation** | Slides, screen display | >= 20pt | >= 14pt | >= 10pt | 150 |

## Criteria

| # | Element | Required | Acceptance Standard |
|---|---------|----------|-------------------|
| 1 | **Coordinate Reference** | Yes | Lat/lon labels on axes or gridlines. |
| 2 | **Scale Bar** | Yes | Labeled distance in appropriate units (m or km). True meters when using a metric projection. |
| 3 | **North Arrow** | Yes | Arrow with "N" label. |
| 4 | **Classification Legend** | If applicable | All color-coded symbols explained. |
| 5 | **Depth/Value Colorbar** | If continuous shading shown | Colorbar with labeled quantity and unit (e.g., "Depth (m)"). |
| 6 | **Neatline Border** | Recommended | Alternating black/white ladder border. Required for poster-tier maps; optional for paper/presentation tier. |
| 7 | **Title** | Yes | Concise; technical details deferred to caption. Font size per sizing tier. |
| 8 | **Figure Caption** | Yes | Renderer-based fully justified, sans-serif, in a dedicated axes region below the plot. Each word's pixel width is measured via `get_window_extent(renderer)`; remaining horizontal space is distributed as equal gaps between words. Last line left-aligned. No orphaned lines. States data source with collection year. Font size per sizing tier. |
| 9 | **Label Legibility** | Yes | Feature labels meet sizing tier minimum. No overlapping labels. |
| 10 | **Resolution/Format** | Yes | PNG at minimum DPI for the declared sizing tier. |
| 11 | **Projection Info** | Yes | Coordinate system explicitly stated in caption or on map (e.g., "UTM Zone 10N", "WGS84"). The PI specifies the projection; it is never assumed. |
| 12 | **Colorblind Safety** | Yes | All color distinctions perceivable under deuteranopia/protanopia. Use Okabe-Ito or equivalent palette. |
| 13 | **Data Provenance** | Yes | Caption or annotation cites the data source and collection/publication year (e.g., "Bathymetry from 1 m AUV survey, MBARI 2025"). |
| 14 | **Multi-Panel Labels** | If multi-panel | Panels labeled consistently (e.g., "(a)", "(b)") at a standard position and font size. |

## Scorecard Template

Copy this table into your project's `specs/map-scorecard.md` and fill it in for each map figure.

```markdown
| Figure | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | Notes |
|--------|---|---|---|---|---|---|---|---|---|----|----|-----|----|----|-------|
| [Figure name] | | | | | | | | | | | | | | | |
```

**P** = Pass, **F** = Fail, **-** = Not applicable

## Notes

- Criteria 1-6 and 11 apply to maps only. Time series figures have their own rubric.
- Caption justification is **renderer-based full justification** for all figure types. The technique: (1) wrap text to fit available width using approximate character widths, (2) for each line except the last, measure each word's pixel width via `get_window_extent(renderer)`, (3) distribute remaining horizontal space as equal gaps between words, (4) left-align the last line. This produces professional typeset-quality captions that fill the full width evenly.
- Neatline (criterion 6) is required at poster tier and recommended at paper/presentation tier. Projects may override in their constitution.
- Projection (criterion 11): the coordinate system must be explicitly identified by the PI before it is used. Never assume a projection — always confirm with the PI first.
- Data provenance (criterion 13): at minimum, name the dataset and its year. For derived products, cite the processing chain or sibling project.
