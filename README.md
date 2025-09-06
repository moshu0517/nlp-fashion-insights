# NLP-Fashion-Insights

A text mining project that applies **TF-IDF** and **NMF** to fashion product descriptions in order to uncover the underlying style themes most relevant to the market. The goal is to help design teams better align new product ideas with customer demand.

## Project Background

In e-commerce, the *sample-to-product ratio* is often low. Designers may create many prototypes, but only a small fraction make it into production because creative intuition does not always match customer preferences.  
This project explores how natural language processing can support decision-making by extracting themes directly from product descriptions, providing a data-driven view of what customers value.

A public dataset is used here for demonstration purposes, but the pipeline reflects what could be applied in a real business setting.

## Methods

- **Text Preprocessing**: cleaning, tokenization, stopword removal, and n-grams  
- **Feature Extraction**: TF-IDF to quantify term importance across thousands of descriptions  
- **Theme Discovery**: NMF to group terms into interpretable style topics  
- **Evaluation**: testing different numbers of themes and refining TF-IDF thresholds

## Key Results

- Identified **8 coherent style themes**, e.g. casual basics, formal wear, athleisure, vintage  
- Represented each product as a weighted mix of discovered themes  
- Observed that theme quality improved with n-gram inclusion and tuned parameters

## Tech Stack

- Python 3.8+  
- scikit-learn  
- pandas / numpy  
- matplotlib

## Repository Structure

## How to Run

```bash
git clone https://github.com/yourusername/nlp-fashion-insights.git
cd nlp-fashion-insights
pip install -r requirements.txt
python fashion_analyzer.py
