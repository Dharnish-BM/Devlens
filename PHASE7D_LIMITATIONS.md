# Phase 7D Limitations

- The GDERS corpus contains only 10 selected repositories.
- The corpus contains 1,225 review comments and is not complete developer activity.
- Category support is uneven; benchmark category slices range from 4 to 18 queries.
- Only 6 multi-category benchmark queries are available.
- Natural-language taxonomy mapping is deterministic keyword mapping, not open-ended semantic understanding.
- Linear SVM decision margins are confidence tiers, not calibrated probabilities.
- The reference labels are single-annotator labels; inter-annotator agreement was not measured.
- Phase 5C used targeted lexical candidate selection, which can introduce selection bias.
- Bot and service-account filtering changes the evaluated candidate population and is intentionally conservative.
- Sparse reviewer evidence leaves many identities without recommendation-eligible evidence.
- Repository selection and review culture introduce domain bias.
- Phase 7B results do not consistently outperform simple evidence-count, PR-diversity, or repository-diversity baselines.
- The historical Phase 7B final report and later generated benchmark artifact contain different metric snapshots; both are preserved and labeled.
- The full GDERS regression suite did not produce a verified completion result in the available environment.
- The complete DevLens suite retains one unrelated pre-existing fork-enrichment test failure.
- DevLens and GDERS measure bounded signals from their respective data; neither establishes universal developer capability.
- Future research may address calibration, broader corpora, multi-annotator labels, and stronger evaluation, but no such work is implemented in Phase 7D.
