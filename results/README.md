# Experimental result data

This directory contains de-identified, analysis-ready numeric and categorical
result data supporting the tables and statistical analyses reported in the
submitted manuscript.

## Detailed result files

- `reference_risk_by_item.csv`: item-level xCOMET reference-risk indicators
  supporting Table 2 and Appendix Table A.1.
- `reference_intervention_by_item.csv`: item-level original, retranslated, and
  keep-higher xCOMET scores supporting Table 5 and Appendix Table B.1.
- `mt_icl_method_scores_by_setting.csv`: model--retriever--language setting
  scores for the methods reported in Tables 3 and 8.
- `quality_relevance_tradeoff_by_setting.csv`: setting-level quality and relevance
  statistics supporting Table 4 and Figure 2.
- `fixed_relevance_intervention_coverage_by_setting.csv`: setting-level
  replacement coverage
  and score changes supporting Table 6 and Appendix Table B.2.
- `fixed_relevance_variant_scores_by_setting.csv`: the four fixed-relevance
  reference variants for each of the 45 settings, supporting Tables 7 and 8.
- `comet22_qe_agreement_by_output.csv`: output-level COMET22 and QE scores
  supporting Table 9. Two empty generations are retained and explicitly marked
  invalid for correlation calculations.

## Reported table files

- `Table_2.csv`: aggregate reference-risk statistics.
- `Table_3.csv`: main MT-ICL results across the reported settings.
- `Table_4.csv`: quality--relevance trade-off across selection methods.
- `Table_5.csv`: reference-intervention quality summary.
- `Table_6.csv`: fixed-relevance intervention coverage by retriever.
- `Table_7.csv`: scores for the four fixed-relevance variants.
- `Table_8.csv`: setting-level robustness statistics for the key comparisons.
- `Table_9.csv`: COMET22--QE agreement by reference-quality group.
- `Appendix_Table_A1.csv`: reference-risk statistics by language and split.
- `Appendix_Table_B1.csv`: full-reference-set intervention changes by
  language and split.
- `Appendix_Table_B2.csv`: intervention coverage by language.

Table 1 reports runtime and scoring configurations rather than experimental
result statistics; the corresponding settings are provided in
`configs/experiment_defaults.yaml`.
