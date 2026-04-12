# Time Series Evaluation Rubric

A shared standard for evaluating time series figures across all projects. Derived from the MISO 2011 Eruption figure (Casper & Diva), the first time series to pass all criteria.

## Sizing Tiers

Absolute font sizes depend on the output medium. Each project constitution must declare which tier applies. (Same tiers as the map rubric.)

| Tier | Use Case | Title | Axis/Caption | Tick Labels | Line Weight | Min DPI |
|------|----------|-------|--------------|-------------|-------------|---------|
| **Poster** | Conference posters, large-format prints | >= 24pt | >= 18pt | >= 16pt | >= 2pt | 600 |
| **Paper** | Journal figures, reports | >= 14pt | >= 10pt | >= 8pt | >= 1pt | 300 |
| **Presentation** | Slides, screen display | >= 20pt | >= 14pt | >= 12pt | >= 1.5pt | 150 |

## Criteria

| # | Element | Required | Acceptance Standard |
|---|---------|----------|-------------------|
| 1 | **Axis Labels** | Yes | Both axes labeled with quantity and unit in parentheses (e.g., "Temperature (°C)"). Bold. Font size per sizing tier. |
| 2 | **Dual-Axis Clarity** | If applicable | Secondary y-axis labeled and color-matched to its data series. Units stated. |
| 3 | **Date Formatting** | Yes | Clean date ticks at regular intervals appropriate to the time span; no crowding or overlap. Horizontal or slightly rotated labels. |
| 4 | **Classification Legend** | Yes | All plotted series identified by color, style, and descriptive label. Framed with high alpha for readability over gridlines. |
| 5 | **Event Annotation** | If applicable | Key events marked with vertical line + text label. Consistent visual language across figures (e.g., red dashed = eruption, gray dotted = cruise/servicing). Bold annotation text. |
| 6 | **Title** | Yes | Concise 1-2 line title identifying the phenomenon, time period, and location. Font size per sizing tier. |
| 7 | **Figure Caption** | Yes | Renderer-based fully justified, sans-serif, in a dedicated axes region below the plot. Each word's pixel width is measured via `get_window_extent(renderer)`; remaining horizontal space is distributed as equal gaps between words. Last line left-aligned. States axis quantities, identifies all data series by name and source, and summarizes the key scientific observation. Font size per sizing tier. |
| 8 | **Temporal Aggregation** | Yes | Aggregation level appropriate to the signal (e.g., daily means, hourly, raw). Stated in the caption or axis label. Data trimmed to the scientifically meaningful window (exclude instrument recovery artifacts, pre-deployment noise, etc.). |
| 9 | **Y-Axis Range** | Yes | Constrained to focus on the signal of interest — not auto-scaled to the full sensor range. |
| 10 | **X-Axis Padding** | Yes | Balanced padding on both sides of the data, proportional to the time span. |
| 11 | **Data Gaps** | Yes | Gaps in the record are visible (no silent interpolation). Excluded intervals noted in the caption if scientifically relevant. |
| 12 | **Grid** | Yes | Light background grid (alpha ~0.3) to aid value reading without obscuring data. |
| 13 | **Spine Weight** | Yes | Plot bounding-box spines meet minimum line weight for the sizing tier. |
| 14 | **Line Weight** | Yes | Data lines meet minimum weight for the sizing tier. Consistent across all series in a figure. |
| 15 | **Resolution/Format** | Yes | PNG at minimum DPI for the declared sizing tier. |
| 16 | **Colorblind Safety** | Yes | All color distinctions perceivable under deuteranopia/protanopia. Use Okabe-Ito or equivalent palette. |
| 17 | **Data Provenance** | Yes | Caption or legend identifies the instrument, data source, and deployment period for every plotted series. |
| 18 | **Multi-Panel Labels** | If multi-panel | Panels labeled consistently (e.g., "(a)", "(b)") at a standard position and font size. |
| 19 | **Layout Spacing** | Yes | Plot area explicitly positioned to reserve space for caption below and title above. No clipping of labels or annotations after export. |

## Scorecard Template

Copy this table into your project's `specs/timeseries-scorecard.md` and fill it in for each time series figure.

```markdown
| Figure | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | Notes |
|--------|---|---|---|---|---|---|---|---|---|----|----|----|----|----|----|----|----|----|----|-------|
| [Figure name] | | | | | | | | | | | | | | | | | | | | |
```

**P** = Pass, **F** = Fail, **-** = Not applicable

## Notes

- This rubric applies to time series figures only. Maps are evaluated separately (see `map-evaluation-rubric.md`).
- Caption justification is **renderer-based full justification** for all figure types, consistent with the map rubric. The technique: (1) wrap text to fit available width using approximate character widths, (2) for each line except the last, measure each word's pixel width via `get_window_extent(renderer)`, (3) distribute remaining horizontal space as equal gaps between words, (4) left-align the last line.
- Temporal aggregation (criterion 8): the correct level depends on the science. Daily means suit multi-month vent temperature records; hourly or raw data may be needed for tidal analysis or event detection. State what was used.
- Data gaps (criterion 11): pandas/matplotlib naturally show gaps when plotting datetime-indexed series with NaN. Do not fill or interpolate unless the method is stated.
- Event annotations (criterion 5): maintain a consistent visual language within a project. Document the convention in the project constitution (e.g., "red dashed = eruption, gray dotted = cruise").
- Data provenance (criterion 17): at minimum, name the instrument and deployment period. For derived data from sibling projects, cite the project and output file.
