# Response to Reviewers

Manuscript: Risk-Calibrated Service-Region Routing with Certified Lazy Motion Evaluation for Cleaning Robots

We thank the reviewer for the detailed and constructive report. The revision keeps the four-page limit and addresses each point inside the paper. The main changes are a precise definition of what an exact query means, the forward and suffix recurrences for the forced-edge bound, an explicit statement of the relative-gap convention and the numerical guard, the missing calibration settings, and a corrected treatment of the holdout counts.

## Comment 1. Meaning of an exact query, state merging and the expansion cap

We agree that the earlier text left the status of a failed query ambiguous. The revision now states that the cost comes from a deterministic oracle and is exact only with respect to that oracle's cost table. For the Hybrid A* oracle, states are merged on a 0.20 m grid with 24 heading bins, each query is capped at 50,000 expansions, and a query that reaches the cap returns an infinite cost. A failed query is treated as an infinite edge cost and is distinguished from a proof of physical infeasibility. The certificate is stated relative to the oracle table, not to the continuous motion optimum. The same boundary is repeated in the abstract, the discussion and the conclusion.

Where: Section III, third paragraph; Section IV-B, where the mixed weights and the certificate use oracle costs; Section VI, first paragraph; Section VII.

**Budget check.** We re-evaluated all 18 Hybrid A* instances edge by edge. Of the 463 failed queries at the 50,000-expansion cap, none reached the cap, so every failure came from an exhausted search queue rather than a truncated one. Re-running the cap-hit edges at 200,000 expansions changed no edge cost, no objective and no route. The cap is therefore not the source of failed queries in these experiments. A failure still means that the discrete oracle found no path, not that the physical transition is impossible, and the paper keeps that distinction. The result file is included as oracle_budget.json.

## Comment 2. Forward and suffix recurrence for the forced-edge bound

The revision gives the two tables explicitly. Equation (8) defines the forward table with its base case, equation (9) defines the suffix table with its base case, and equation (10) gives the conditional bound for a cross edge. The text states the compatible subset conditions, namely that the prefix set contains the source group and excludes the destination group, and it gives the start-edge case separately with an empty prefix. Infinite costs use extended arithmetic, with x plus infinity equal to infinity and an empty minimum equal to infinity. The construction covers asymmetric costs and one-of-many group selection.

Where: Section IV-D, equations (8), (9) and (10), and the surrounding text.

## Comment 3. Relative guarantee, UB equal to zero, and the numerical guard

The stopping test is now written as UB minus LB no larger than eta times UB plus tau, with tau equal to 10 to the minus 10. Equation (6) states the relative bound and explains that division by UB is a convention because UB is known and positive while the optimum is not. The revision states that if UB equals zero then the optimum is zero and the procedure reports the absolute gap. The guard is described as part of the stated certificate. The screening test uses the same guard, which relaxes the guarantee by that additive amount. The earlier phrase about machine precision has been replaced by a separate implementation check, with the certificate itself stated up to the guard.

Where: Section IV-B, equation (6) and the following sentences; Section IV-D, last sentence; Section V-B, first paragraph; Section VI, first paragraph.

## Comment 4. Calibration and algorithm settings

The revision specifies the trial counts and the disturbance model. In the Hybrid A* suite each candidate received 220 calibration trials under target noise of 0.05 to 0.09 m, base-position noise of 0.04 m, heading noise of 2 degrees, reach limits of 0.45 to 1.05 m, a heading tolerance of 18 degrees, and a clearance test based on the 0.10 m footprint. The calibration sweep used 240 trials and 2500 holdout trials per candidate, 56 candidates per family, and four target uncertainty levels. The text also states that the confidence family contains all 56 candidates generated before any selection, including candidates that are later rejected, so the family is not defined after route-dependent selection.

Where: Section V-A, third paragraph.

## Comment 5. Holdout uncertainty and the meaning of false acceptance

The revision no longer treats holdout labels as ground truth. The holdout counts are now shown in Fig. 2, split into holdout failures and holdout passes, and the caption states that the counts are observations from an independent sample. For geometric acceptance, 595 of 896 candidates fell below the 0.85 threshold on the holdout, an observed rate of 66.4 percent with a 95 percent Wilson interval of 63.2 to 69.4 percent. For the two calibrated rules, the revision reports the one-sided 97.5 percent Clopper-Pearson upper bound on the failure rate, 1.6 percent for 223 retained candidates and 2.0 percent for 180. The text states that a zero count is not evidence of zero risk. The earlier phrase about known false acceptance or rejection has been removed.

Where: Section V-D and Table II.

## Other changes required by the page limit

The revision keeps the four-page limit. To make room for the added material, the four-suite summary table was removed and its numbers are now given in one sentence in Section V-B. The related-work section was compressed to a single paragraph. The real-map breakdown and the holdout counts remain in the paper, now as Fig. 1 and Fig. 2. The adaptively screened baseline is now reported in text rather than in a separate table. No result was changed, and all numbers still come from the same result files as the earlier version.

## Summary of locations

| Comment | Main location |
|---|---|
| 1 Oracle semantics | Section III, para. 3; Sections IV-B, VI, VII |
| 2 Recurrences | Section IV-D, equations (8) to (10) |
| 3 Gap and guard | Section IV-B, equation (6); Section IV-D; Section V-B |
| 4 Calibration settings | Section V-A, para. 3 |
| 5 Holdout uncertainty | Section V-D; Fig. 2 |
