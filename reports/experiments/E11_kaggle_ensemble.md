# E11 (Q7) Ensemble (Data - Round 2)

(Q7 in the docs)

RQ: Given a fixed set of trained Re-ID models, do different fusion strategies exploit complementary errors and improve retrieval over the best single model, and does that depend on the gallery protocol?

RQ: Can ensemble fusion improve jaguar Re-ID over the best single model by exploiting complementary per-query strengths, and how robust are these gains across gallery protocols?




Across the internal evaluation protocols, fusion methods consistently outperform the best single model, indicating useful complementarity between model errors. However, the gains do not necessarily transfer unchanged to the Kaggle hidden test set, suggesting that part of the fusion advantage may be protocol-specific rather than fully robust out-of-sample.