# AlphaRAG: Institutional Knowledge Extractor

Welcome to **AlphaRAG**, an enterprise-grade Retrieval-Augmented Generation system for financial document analysis.

## Capabilities

-  **Precise data extraction** from SEC filings (10-K, 10-Q, 8-K)
-  **Grounded answers** — all responses are sourced exclusively from indexed documents
-  **Interactive citations** — click any source footnote to view the exact retrieved passage
-  **Parent-Child retrieval** — dense child chunks for search precision, broad parent chunks for context richness

## How to Use

1. Type your question in the chat box below
2. Review the answer and its source citations
3. Click any `Source N` link to open the full retrieved passage in the sidebar
4. If you see *"Insufficient data in the provided filings"*, the answer is not in the indexed documents

## Example Queries

- *What was [Company]'s total revenue and gross margin for FY2025?*
- *What are the key risk factors related to competition and supply chain?*
- *How did operating expenses change year-over-year?*
- *What guidance did management provide for the upcoming fiscal year?*

---

*Powered by Groq (LLaMA-3 70B) · BAAI/bge-m3 Embeddings · Qdrant Vector Store*
