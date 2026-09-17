# Evaluation of PaperMindAI Strategies

This document presents an evaluation comparing our distinct chunking and retrieval strategies.

## Overview
We built PaperMindAI and provided configurable chunking and retrieval strategies. To properly evaluate the RAG pipeline, we queried 10 unique topics across multiple PDF documents and reviewed the responses.

### Chunking Strategies
- **Strategy A (size = 1000, overlap = 200)**: This granularity isolates specific topics and yields smaller but very focused context blocks.
- **Strategy B (size = 1500, overlap = 300)**: This captures broader context in a single chunk, which is helpful if answers require reading full paragraphs or multiple sequential concepts.

### Retrieval Strategies
- **Similarity (Top K=3)**: Focuses strictly on cosine proximity to the embedding query. Can lead to retrieving 3 adjacent chunks containing repetitive information.
- **MMR - Max Marginal Relevance (Top K=3)**: Operates by penalizing chunks that are too similar to already selected ones. Increases semantic diversity spanning across the paper.

---

## 10 Evaluation Queries & Observations
Below are the 10 specific queries used during testing and generalized observations from our pipeline.

### 1. Simple Fact Retrieval
- **Query 1**: "What is the proposed baseline setup?"
- **Observation**: Strategy A (1000/200) and Similarity Search performed exceptionally well here, finding the specific snippet and quickly yielding an accurate answer. MMR occasionally pulled in less relevant background paragraphs simply to enforce diversity.

### 2. Multi-Hop / Complex Concept Synthesis
- **Query 2**: "How does the proposed architecture differ from prior approaches mentioned in the literature review?"
- **Observation**: Strategy B (1500/300) was crucial. Strategy A often cut-off context mid-paragraph, meaning the LLM could see the proposed architecture but not the contrasting prior approaches. Further, **MMR Retrieval significantly outperformed Similarity**. MMR pulled chunks from the 'Architecture' section and the 'Literature Review' section, avoiding repetitive focus on a singular paragraph.

### 3. Missing Information Check (Hallucination Test)
- **Query 3**: "What specific GPU hardware was used to train the model?" (When the paper omits this detail)
- **Observation**: The RAG pipeline responded dependably with "I don't know" under all strategies. This validates the strict system prompt.

### 4. Technical Definition Lookup
- **Query 4**: "Define the term 'Cross-Entropy Loss' as used in the paper."
- **Observation**: Clean extraction under Strategy A + Similarity.

### 5. Equation/Formula Retrieval
- **Query 5**: "What is the formula used for calculating attention weights?"
- **Observation**: Both strategies performed well, but the PyMuPDF4LLM extraction successfully preserved markdown formatting for the equation.

### 6. Quantitative Results Check
- **Query 6**: "What was the highest achieved accuracy on the benchmark dataset?"
- **Observation**: Strategy A was best. The LLM accurately cited the corresponding table from the text context.

### 7. Core Contribution Summary
- **Query 7**: "Summarize the three main contributions of this research."
- **Observation**: Strategy B + MMR dominated here. MMR forced the retriever to grab from the abstract, intro, and conclusion sections instead of getting stuck on one paragraph.

### 8. Dataset Specifics
- **Query 8**: "How many training samples were included in the synthetic dataset?"
- **Observation**: Both methods easily pinpointed the data.

### 9. Out-of-Scope Factual Query
- **Query 9**: "Who won the World Cup in 2022?"
- **Observation**: Model refused to answer based on system prompt directive 4.

### 10. Citation Formatting Verification
- **Query 10**: "What does the author conclude about future work?"
- **Observation**: Regardless of strategy, injecting explicitly parsed `source` and `page` metadata alongside every chunk inside the LLM context allowed consistent and structured citation outputs. The model did not hallucinate page numbers.

---

## Embeddings Comparison (Open-Source vs Commercial)

As part of our architectural validation, we built a UI toggle allowing instant switching between two distinct embedding models, indexing the exact same PDFs using both systems for comparison.

- **HuggingFace (`BAAI/bge-small-en`) [Open-Source]**: As a completely local baseline, `bge-small-en` effectively captured standard keyword density and performed flawlessly on semantic proximity. Best of all, it incurs zero API cost and runs entirely offline.
- **Google Gemini (`models/text-embedding-004`) [Commercial]**: Leveraging Google's state-of-the-art embedding endpoint yielded noticeably stronger performance at capturing abstract context and "dense logic" across multi-hop scenarios. Because Gemini maps embeddings across a highly complex semantic space, ambiguous queries were matched to specific architectural paragraphs vastly quicker than the local model.

---

## Conclusion
- **For heavily technical or scattered contexts**: Use **Strategy B + MMR**.
- **For direct empirical lookups (like definitions or exact constants)**: Use **Strategy A + Similarity** for the quickest and most precise focus.

*PaperMindAI satisfies the ability to evaluate these strategies actively in the application's sidebar.*
