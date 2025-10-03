# scQC Agent Knowledge Base

This directory contains curated knowledge for the scQC Agent's RAG system.

## Contents

- `qc_guidelines.md` - Quality control best practices and thresholds
- `scanpy_qc_docs.md` - Scanpy QC function documentation
- `doublet_detection.md` - Doublet detection methods and parameters
- `batch_integration.md` - Batch correction and integration strategies
- `graph_analysis.md` - Graph construction and clustering guidance
- `workflows/` - Standard operating procedures
- `thresholds/` - Species-specific and tissue-specific recommendations

## Usage

The RAG system automatically ingests these documents and uses them to provide
context-aware responses to user queries about scRNA-seq QC workflows.

## Organization

Documents are organized by topic and include metadata for retrieval:
- `topic`: Main category (quality_control, doublets, integration, etc.)
- `species`: Applicable species (human, mouse, general)
- `tissue`: Tissue type if specific
- `method`: Computational method or tool
