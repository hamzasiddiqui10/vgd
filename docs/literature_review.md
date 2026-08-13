# Literature Review: Video Action Generalization

| Paper Title | Base Dataset | Scale | Classes | Domain Shift Tested | Non-Object Shift Suitability | Verb/Motion Gap Filled |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Something-Something v2** (Goyal et al.) | Something-Something v1 | 220,847 clips (approx. 215 hrs) | 174 unique classes (Motion-based / Fine-grained verbs) | Unseen objects, background variations, and daily physical interactions. | **High:** Isolates motion by pairing similar verbs with vastly different everyday objects, preventing models from relying on static scenes. | **Covers core gap:** Provides an excellent template for fine-grained verb control, but lacks explicit cross-view or camera-motion shifts. |
