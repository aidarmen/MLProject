# Week 4 — Deployment

Streamlit app for the Airline Passenger Satisfaction model trained in Week 3.

## Files

| File | Purpose |
|------|---------|
| `save_model.py` | Trains the XGBoost pipeline on the full training set and saves `pipeline.joblib` + `feature_meta.json` |
| `app.py` | Streamlit app — input form, prediction, SHAP waterfall |
| `requirements.txt` | Python dependencies for Streamlit Community Cloud |
| `ablation_study.ipynb` | Mandatory ablation experiments for the technical report |
| `Group5_Week4_Technical_Report.docx` | Journal-style technical report |

## Local run

```bash
pip install -r requirements.txt
python save_model.py            # writes pipeline.joblib and feature_meta.json
streamlit run app.py
```

Open http://localhost:8501 in a browser.

## Deploy to Streamlit Community Cloud

1. Create a new public GitHub repo and push the contents of this `week4/` folder
   (including `pipeline.joblib` and `feature_meta.json` after running `save_model.py`).
2. Visit https://share.streamlit.io and click **New app**.
3. Select the repo, branch, and `app.py` as the main file.
4. Click **Deploy**. The first build takes ~3 minutes.
5. Test from a different machine and network at least 48 hours before the presentation.

## Deploy to Hugging Face Spaces (alternative)

1. Create a new Space with the **Streamlit** SDK.
2. Upload `app.py`, `requirements.txt`, `pipeline.joblib`, `feature_meta.json`.
3. The Space builds and serves automatically.
