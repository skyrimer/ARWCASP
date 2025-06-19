# 4CBLW00-20: Addressing Real-World Crime and Security Problems with Data Science

An automated, data-driven police demand forecasting system to inform resource allocation for residential burglary prevention in London, UK.


## Table of Contents

- [Project Overview](#project-overview)  
- [Problem Description](#problem-description)  
- [Software & Resources](#software--resources)  
- [Project Structure](#project-structure)  
  - [📁 Root Directory](#📁-root-directory)  
  - [📁 EDA/](#📁-eda)  
  - [📁 data/](#📁-data)  
  - [📁 data_download_pipeline/](#📁-data_download_pipeline)  
  - [📁 processed_data/](#📁-processed_data)  
  - [📁 model/](#📁-model)  
  - [📁 routes/](#📁-routes)  
  - [How It All Fits Together](#how-it-all-fits-together)
- [Running the dashboard](#running-dashboard)
- [Authors](#authors)  
- [References](#references)  
  - [Official Statistics & Government Data](#official-statistics--government-data)  
  - [Academic & Industry Research](#academic--industry-research)  
  - [Technical Tools & Libraries](#technical-tools--libraries)  
  - [Licensing & Terms](#licensing--terms)  


## Project Overview

Despite COVID changing the landscape of crime in the United Kingdom, residential burglaries persist to be a problem with over 250,000 offences in 2024. Police's attempts to combat the crime is challenged by limited resources, budgetary constraints, and the need for strategic allocation of patrol efforts. Throughout our research we have found the Dutch AI predictive policing models resulted in meaningful prevention of high‐impact crimes via automated patrol guidance as well as focused interventions at crime hotspots produced small but significant decreases in burglary and other property crimes. We have also found historical data of residential burglaries were studies without aiming to predict future ones, and thus police has to rely on CCTV positioning improvements to combat the crime.

Therefore, based on the before-mentioned research and our desire to help tackling burglaries, our group developed a 3-step data-driven tooling that empowers law enforcement with future and prevention focused insights


## Problem Description

- **Domain:** Allocation of police resources to prevent and respond to residential burglary in London, UK.  
- **Research Question:**  
  > How can we best estimate police demand in an automated manner to inform the most effective use of police resources to reduce residential burglary in London (UK)?
- **Objective:**  
  1. Create a predictive model for burglary counts per LSOA for upcoming months 
  2. Develop automated data-driven police routes based on predictions  
  3. Design and implement an interactive dashboard that enables clear and concise presentation of above-mentioned systems for police leadership.  

## Software & Resources
- **Main programming language:** Python
- **Core Libraries:**  
  - Data retrieval: osmnx, requests
  - Data Wrangling: pandas, 
  - Geospatial data handling: shapely, folium, geopandas  
  - Modeling & Forecasting: pytorch, pyro, optuna
  - Routing algorithm: osmnx, python_tsp, networkx, shapely
  - Dashboard/Visualization: Matplotlib, Plotly, Streamlit  
- **Version Control & Collaboration:** Git, GitHub  
- **Documentation & Reporting:** Latex report

## Project Structure

### 📁 Root Directory

- **README.md**  
  High-level project overview, problem statement, and setup instructions.

- **requirements.txt**  
  Python dependencies installation list (`pip install -r requirements.txt`).

- **Merge.ipynb**  
  Jupyter notebook that merges and cleans raw data into `merged_data.parquet`.

- **merged_data.parquet**  
  Consolidated, cleaned dataset for modeling.

- **dashboard.py**  
  Launches the interactive dashboard (Streamlit/Dash) to visualize predictions and routes.

- **Data Description-2.pdf**  
  Detailed schemas and descriptions of all raw data sources.

- **.gitignore**  
  Specifies files/directories to exclude from version control.


### 📁 `EDA/`

Contains exploratory analysis notebooks, for example:

- **basic_eda.ipynb**  
  Visualizes burglary trends (time series, seasonality) with Matplotlib and Seaborn.


### 📁 `data/`

Holds manually downloaded raw data:

- Police-recorded crime CSVs (2010–2025)  
- Shapefiles for LSOAs, wards  
- IMD Excel files (2015, 2019)  
- LSOA code-lookup tables


### 📁 `data_download_pipeline/`

Automated data-fetch scripts:

- **download_baseline.py**  
  Downloads burglary records via the UK police API.

- **openstreetmapdownload.py**  
  Uses **OSMnx** to fetch POIs and road networks from OpenStreetMap.


### 📁 `processed_data/`

Stores intermediate cleaned tables (Parquet/CSV) produced by `Merge.ipynb`, ready for feature engineering.


### 📁 `model/`

Implements the Bayesian spatio-temporal burglary model and training pipeline:

- **model.py**  
  Defines `burglary_model` in Pyro for hierarchical Poisson regression.

- **pipeline.py**  
  Data preparation (`prepare_model_data`), SVI training, and cross-validation routines.

- **train_and_evaluate.py**  
  Wrapper to train on one split, evaluate RMSE/MAE/CRPS, and return results.


### 📁 `routes/`

Generates optimized patrol routes using predicted hotspots:

- **borough_routes.py**  
  Creates borough-level waypoints and solves a TSP via **NetworKit** heuristics.

- **ward_routes.py**  
  Similar to borough routing but at ward granularity.

- **utils.py**  
  Helper functions for OSMnx boundary processing and edge-cost definitions.


### How It All Fits Together

1. **Data Acquisition**: Run scripts in `data_download_pipeline/` to fetch raw sources.  
2. **Data Merging**: Execute `Merge.ipynb` → `merged_data.parquet`.  
3. **Exploration**: Inspect trends in `EDA/basic_eda.ipynb`.  
4. **Feature Engineering**: Use `processed_data/` tables for covariate creation.  
5. **Modeling**: Train & validate Bayesian model in `model/`.  
6. **Routing**: Generate patrol paths under `routes/`.  
7. **Dashboard**: Launch `dashboard.py` for interactive visualization.

## Running the dashboard
To run the dashboard, open up a terminal and run the following command in the root folder of the GitHub:
```
streamlit run dashboard.py
```
This will open up your webbrowser with the dashboard running there. For more information about the dashboard see the report.

## Authors
- Kirill Chekmenev
- Neha Joseph
- Rayan Mahmoud
- Elisa Martín Morán
- Joshua Pantekoek
- Diederik Webster

## References

### Official Statistics & Government Data
- **Office for National Statistics**. *Crime in England and Wales: year ending December 2024*. ONS, 24 April 2025.  
  https://www.ons.gov.uk/peoplepopulationandcommunity/crimeandjustice/bulletins/crimeinenglandandwales/yearendingdecember2024

- **data.police.uk**. *Police Recorded Crime & ASB Incidents* (Dec 2010–Mar 2025). UK Police Data.  
  https://data.police.uk/data

- **London Datastore**. *Statistical GIS Boundary Files – LSOA & Ward Shapefiles*.  
  https://data.london.gov.uk/dataset/statistical-gis-boundary-files-london

- **Consumer Data Research Centre (CDRC)**. *Index of Multiple Deprivation (IMD) Datasets* (2015 & 2019).  
  https://data.cdrc.ac.uk/dataset/index-multiple-deprivation-imd

- **Ministry of Housing, Communities & Local Government (MHCLG)**. *English Indices of Deprivation 2019 – Mapping Resources*. (Modified 24 Apr 2025)  
  https://www.gov.uk/guidance/english-indices-of-deprivation-2019-mapping-resources

- **Open Data London**. *LSOA Atlas: Dwelling Types, Ethnicity, Households, Population, Tenure, PTAL* (2011 & 2013).  
  https://data.london.gov.uk/dataset/lsoa-atlas

- **Office for National Statistics**. *Lower-layer Super Output Areas (Dec 2021) Boundaries*.  
  https://geoportal.statistics.gov.uk/datasets/ons::lower-layer-super-output-areas-december-2021-boundaries-ew-bsc-v4-2

- **data.gov.uk**. *LSOA 2011→2021 Code Lookup (Exact Fit)*.  
  https://data.gov.uk/dataset/ons::lsoa-to-lsoa-exact-fit-lookup

### Academic & Industry Research
- den Heyer, G. (2014). “Examining Police Strategic Resource Allocation in a Time of Austerity.” *Salus Journal: An International Journal of Law Enforcement & Public Safety*, 2(1), 63–79.  
  https://salusjournal.com/wp-content/uploads/2013/03/den_Heyer_Salus_Journal_Issue_2_Number_1_2014_pp_63-79.pdf

- Browning, C. R. et al. (2010). “Commercial Density, Residential Concentration, and Crime: Land Use Patterns and Violence in Neighborhood Context.” *Journal of Research in Crime and Delinquency*, 47(3), 329–357.  
  https://doi.org/10.1177/0022427810365906

- Storbeck, M., & EUCPN Secretariat. (2022). *Artificial Intelligence and Predictive Policing: Risks and Challenges* (Recommendation Paper No. 2). European Union Crime Prevention Network.  
  https://eucpn.org/sites/default/files/document/files/PP%20%282%29.pdf

- Braga, A. A., & Weisburd, D. L. (2020). “Does Hot Spots Policing Have Meaningful Impacts on Crime?” *Journal of Quantitative Criminology*, 38, 1–22.  
  https://cina.gmu.edu/wp-content/uploads/2020/11/Braga-Weisburd2020_Article_DoesHotSpotsPolicingHaveMeanin.pdf

- Halford, E., & Gibson, I. (2025). “Using Machine Learning to Conduct Crime Linking of Residential Burglary.” *International Journal of Law, Crime and Justice*, 80, 1–15.  
  https://www.sciencedirect.com/science/article/pii/S1756061624000685

- Reese, H. (2022, Feb 23). “What Happens When Police Use AI to Predict and Prevent Crime?” *JSTOR Daily*.  
  https://daily.jstor.org/what-happens-when-police-use-ai-to-predict-and-prevent-crime/

### Technical Tools & Libraries
- **OpenStreetMap Contributors**. *OpenStreetMap* (ODbL).  
  https://www.openstreetmap.org

- Boeing, G. (2017). “OSMnx: New Methods for Acquiring, Constructing, Analyzing, and Visualizing Complex Street Networks.” *Journal of Open Source Software*  
  https://github.com/gboeing/osmnx

- **NetworKit Developers**. *NetworKit: Large-Scale Network Analysis Toolkit*.  
  https://networkit.github.io

- **Folium Contributors**. *Folium: Python Data. Viz. on Leaflet Maps*.  
  https://python-visualization.github.io/folium

### Licensing & Terms
- **Open Government Licence v3.0**. UK Government.  
  http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/

- **Open Data Commons Open Database License (ODbL)**. Open Data Commons.  
  https://opendatacommons.org/licenses/odbl/

