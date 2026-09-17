# Machine Learning Based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence

## Project
Prepared for the Unified Mentor project brief. The supplied PRD asks for data cleaning, encoding, scaling, K-Means and hierarchical clustering, cluster validation, interpretation, a research paper and a Streamlit dashboard.

## Files
- `Research_Paper_Real_Estate_Buyer_Segmentation.docx` — editable research paper.
- `Research_Paper_Real_Estate_Buyer_Segmentation.pdf` — submission-ready PDF.
- `app.py` — Streamlit dashboard.
- `buyer_segmentation_results.csv` — cleaned client-level analysis with cluster assignments.
- `transactions_cleaned.csv` — cleaned transaction data with numeric sale price and derived fields.
- `Real_Estate_Buyer_Segmentation_Model.xlsx` — model outputs, QA, evaluation and buyer-level results.
- `requirements.txt` — Python dependencies.

## How to run the dashboard
1. Install Python 3.10+.
2. In this folder run: `pip install -r requirements.txt`
3. Run: `streamlit run app.py`

## Modeling note
The supplied dataset does not contain income, investment amount, net worth or explicit property investment value per buyer. Therefore, the analysis does **not** claim to identify “high-income” or “large-investment” buyers. Instead, it uses observable transaction activity, transaction value, property size/category mix, financing, purpose, demographics and acquisition channel.

The supplied PRD lists four recommended archetypes as an example. The actual data-driven validation produced K=3 as the highest-silhouette solution among K=2..8, so the final segmentation uses 3 clusters.

## Data quality
Clients: 2,000 rows; no duplicate rows and no missing cells in the supplied client table.
Transactions: 10,000 rows; 7,305 Sold and 2,695 Available. `client_ref` is blank/unmatched for 2,695 rows; these are retained because they represent available listings without a linked buyer.
