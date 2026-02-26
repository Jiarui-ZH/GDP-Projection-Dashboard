# GDP Projection Dashboard

An interactive web dashboard for long-run GDP forecasting built on the **Conditional Convergence Model (CCM)**. The dashboard lets you explore projections for 100+ countries out to 2050, adjust model parameters in real time, and export results to Excel.

---

## What the Model Does

The CCM projects a country's GDP using three drivers:

1. **Productivity convergence** — countries tend to converge toward a long-run steady state (LSS) relative to US productivity. The gap closes at rate β each year.
2. **US frontier growth** — the global technology frontier (proxied by US trend growth, `g_usa`) lifts all countries over time.
3. **Demographic dividend** — the working-age population (WAP) ratio captures how demographic shifts accelerate or drag on growth.

The core formula for GDP per capita in year *t*:

```
GDPPC_t = GDPPC_base
          × (RelProd_t / RelProd_base)    ← productivity catch-up
          × exp(g_usa × Δt)              ← frontier growth
          × (WAPratio_t / WAPratio_base)  ← demographic effect
```

The **long-run steady state** for each country is estimated via **Gaussian kernel regression** on Global Competitiveness Index (GCI) scores — countries with higher institutional and structural quality converge to a higher productivity level relative to the US.

Total GDP (billions, 2021 USD PPP) is then: `GDPPC × Population (thousands) / 1,000,000`.

---

## Dashboard Features

| Tab | Description |
|---|---|
| **Total GDP** | Country-level GDP trajectories (trillions), with historical data from 1990 and projections to 2050 |
| **GDP per Capita** | GDPPC paths for selected countries, showing convergence dynamics |
| **Regional GDP** | Aggregated GDP by region (ASEAN, Europe, Latin America, etc.) and G7 comparisons |
| **Productivity Convergence** | Visualises each country's path toward its LSS, with the kernel regression curve |
| **Presentation Charts** | Publication-ready charts for reports and slides |
| **Technical Details** | Model equations, parameter sensitivity, and methodology notes |
| **Export** | Download full forecast data as a formatted Excel file |

**Sidebar controls** let you tune the model interactively:
- **β (beta)** — convergence speed (how fast countries close the productivity gap)
- **g_usa** — assumed US long-run growth rate
- **Kernel bandwidth** — smoothness of the GCI → LSS relationship
- **GCI weights** — relative weights of GCI sub-components (A, B, C)

---

## Running the Dashboard Locally

### Prerequisites

- Python 3.10+
- The Excel model file at `model/Version 6.xlsx`

### Install dependencies

```bash
pip install -r requirements.txt
```

### Launch

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

---

## Deploying & Sharing

The easiest way to share the dashboard publicly is via **Streamlit Community Cloud** (free):

1. Push this repo to GitHub (make sure `model/Version 6.xlsx` is included)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with your GitHub account
4. Click **New app** → select this repo → set the main file to `app.py`
5. Click **Deploy** — Streamlit will build and host the app at a public URL you can share

> **Note:** The Excel model file must be committed to the repo for the cloud deployment to work, since the app reads it directly on startup.

---

## Project Structure

```
GDP-Projection-Dashboard/
├── app.py                  # Streamlit dashboard
├── requirements.txt        # Python dependencies
├── model/
│   └── Version 6.xlsx      # Underlying Excel model (data + parameters)
└── utils/
    ├── data_loader.py      # Reads and parses all Excel sheets into DataFrames
    └── model.py            # CCM projection engine (convergence, kernel, aggregation)
```
