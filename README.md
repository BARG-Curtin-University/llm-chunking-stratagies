# LLM-Chunking-Strategies

This repository contains the source code and materials for the white paper
titled "Optimising Language Models with Advanced Text Chunking Strategies". The
paper explores various techniques for enhancing the performance and accuracy of
large language models (LLMs) through advanced text chunking strategies and text
embeddings, with a particular focus on retrieval-augmented generation (RAG)
applications.

## Overview

The white paper delves into the following key topics:

- **Retrieval-Augmented Generation (RAG)**: An in-depth analysis of the RAG
  framework, which enables language models to search through external, dynamic
  knowledge bases and incorporate relevant information into their responses.
- **Text Embeddings**: The role of text embeddings in converting textual data
  into numerical representations that capture semantic meanings, facilitating
  computational processing and analysis.
- **Text Chunking Strategies**: A comprehensive exploration of various text
  chunking techniques, ranging from basic character splitting to advanced
  methods like semantic chunking, agentic chunking, and the subdocument RAG
  technique.
- **Subdocument RAG Technique**: A detailed examination of the subdocument RAG
  technique, which leverages document summaries to improve retrieval efficiency
  and contextual relevance.
- **Future Research Directions**: Potential areas for further exploration,
  including multimodal embeddings, unsupervised chunking, adaptive chunking,
  incremental knowledge base updates, explainable chunking and retrieval, and
  integration with knowledge graphs.

## Repository Structure

The repository is organised as follows:

```
LLM-Chunking-Strategies/
├── data/
│   └── ... (datasets and knowledge bases used in the paper)
├── src/
│   └── ... (source code for implementing the proposed techniques)
├── _quarto.yml
├── index.qmd (the main Quarto manuscript file)
├── README.md
└── references.bib
```

## Getting Started

To get started with this repository, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/LLM-Chunking-Strategies.git`
2. Install the required dependencies (e.g., Python packages, Quarto, etc.) by following the instructions in the `requirements.txt` file.
3. Explore the `src/` directory for the implementation of the proposed text chunking strategies and RAG techniques.
4. Run the experiments and evaluations using the scripts and notebooks in the `experiments/` directory.
5. Refer to the `paper/paper.qmd` file for the Quarto manuscript source code and compile it to generate the final white paper.

## Contributing

Contributions to this repository are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request. When contributing, please follow the guidelines outlined in the repository's `CONTRIBUTING.md` file.

## Citation

If you use the techniques or findings from this white paper in your research or work, please cite the paper as follows:

```bibtex
@article{borck2024optimising,
  title={Optimising Language Models with Advanced Text Chunking Strategies},
  author={Borck, Michael},
  journal={arXiv preprint arXiv:...},
  year={2024}
}
```

## License

This repository is licensed under the [MIT License](LICENSE).

## Acknowledgements

We would like to thank the researchers and developers whose work has contributed to the advancement of natural language processing and language model optimisation techniques.