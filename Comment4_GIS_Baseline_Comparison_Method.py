"""
Reproducible GIS baseline comparison for Reviewer 1, Comment 4.

Inputs:
- Export_Output_3.shp
- generalized_geology.gpkg
- copper_points_clustered.gpkg
- Comment4_Step1_Semantic_Baseline_Mappings.xlsx

Protocol:
1. Make source and tectonic geometries valid and two-dimensional.
2. Reconstruct the 11 common tectonic-unit footprints from generalized_geology.gpkg.
3. Intersect the original geological polygons with this common partition once.
4. Attach each semantic baseline label to the same partition.
5. Dissolve by TectonicUnit + MethodLabel.
6. Calculate feature count, polygon parts, vertices, summed perimeter,
   topology diagnostics, copper linkage, and measured runtime.
7. Report common partition construction separately from method-specific runtime.

The workflow intentionally does not invent or retrospectively estimate manual labor time.
See Comment4_Step2_GIS_Baseline_Comparison.xlsx for results and protocol details.
