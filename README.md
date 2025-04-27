# 4CBLW00-20: Addressing Real-World Crime and Security Problems with Data Science

An automated, data-driven police demand forecasting system to inform resource allocation for residential burglary prevention in London, UK.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Problem Description](#problem-description)  
- [Background Information](#background-information)  
- [Expectations](#expectations)  
- [Resource Limitations](#resource-limitations)  
- [Data Sources](#data-sources)  
- [Accessing the Data](#accessing-the-data)  
- [Background Reading](#background-reading)  
- [Software & Resources](#software--resources)  

---

## Project Overview

Law enforcement agencies and national policymakers face rapidly changing crime patterns and budget constraints. Your team’s mission is to develop an automated, data-driven forecasting system that predicts police demand for residential burglary in London. This tool will help decision-makers allocate officers at the right place and time, while carefully considering ethical implications.

---

## Problem Description

- **Domain:** Allocation of police resources to prevent and respond to residential burglary in London, UK.  
- **Objective:**  
  1. Forecast short- and long-term burglary demand at ward level.  
  2. Recommend optimal spatial and temporal deployment of officers.  
  3. Reflect on ethical considerations in automated decision support.  
- **Key Question:**  
  > How can we best estimate police demand in an automated manner to inform the most effective use of police resources to reduce residential burglary in London?

---

## Background Information

Burglary prevention and response strategies typically include:  
1. **Rapid Response:** Arrive quickly to catch offenders in the act.  
2. **Detective Work:** Use forensics, CCTV, and witness identification post-incident.  
3. **Preventative Measures:** Increase visible patrols, target hardening, and offender diversion.

An accurate understanding of burglary patterns—when, where, how, and why—enables more effective deployment of these strategies. In densely populated, high-value areas like London, finite officer resources make burglary a “wicked problem” despite well-known tactics.

---

## Expectations

- **Forecasts:**  
  - Short-term (daily/weekly) and longer-term (seasonal/multi-year) burglary demand by ward.  
  - Clear recommendations for patrol schedules and hotspot targeting.  
- **Evidence Base:**  
  - Leverage crime-reduction literature (e.g., What Works Centre for Crime Reduction).  
  - Justify methodological choices against known best practices.  
- **Ethical Analysis:**  
  - Address potential biases (e.g., over-policing in deprived areas).  
  - Propose safeguards for transparency and accountability.

---

## Resource Limitations

- **Geography:** London is divided into boroughs and wards.  
- **Officer Availability:**  
  - 100 officers per ward, patrol hours 06:00–22:00.  
  - Only 2 hours/day, 4 days/week dedicated to burglary prevention (200 officer-hours/day).  
  - Special operations (additional officers) allowed once every 4 months.  
- **Movement Constraint:** Officers cannot cross ward boundaries.

---

## Data Sources

1. **Police-Recorded Crime Data (Dec 2013–Present)**  
   - All reported crimes in England, Wales, and Northern Ireland (>70 million cases).  
   - Aggregated monthly and to LSOA level for privacy.
2. **Spatial Boundaries**  
   - LSOA and ward shapefiles from London Datastore (Statistical GIS Boundary Files).  
   - Police force/neighborhood boundaries from police.uk.
3. **Additional Contextual Data (Optional)**  
   - IMD – Index of Multiple Deprivation (ONS).  
   - Societal well-being indices, demographic data, weather, etc.

---

## Accessing the Data

- **Crime & ASB Incidents:**  
  https://data.police.uk/data  
- **Stop-and-Search & Outcomes:**  
  https://data.police.uk/data  
- **Boundary Shapefiles:**  
  https://data.london.gov.uk/dataset/statistical-gis-boundary-files-london  
  https://data.police.uk/data/boundaries  
- **Deprivation Indices:**  
  https://opendatacommunities.org/def/concept/folders/themes/societal-wellbeing  
  https://www.gov.uk/guidance/english-indices-of-deprivation-2019-mapping-resources  

---

## Background Reading

- **Anonymisation Procedure:** police.uk Data About Page  
- Tompson _et al._ (2014). *UK open source crime data: accuracy and possibilities for research*.  
- Laufs _et al._ (2021). *Understanding the concept of ‘demand’ in policing* (scoping review).  
- Hyndman & Athanasopoulos (2018). *Forecasting: Principles & Practice* (R-based, but methods apply in Python).

---

## Software & Resources

- **Recommended Platforms:** Python, R, Jupyter, GitHub  
- **Core Libraries:**  
  - Data Wrangling: pandas, geopandas, SQLAlchemy  
  - Modeling & Forecasting: scikit-learn, statsmodels, fbprophet/NeuralProphet  
  - Geospatial: shapely, folium, keplergl  
  - Visualization: Matplotlib, Plotly, Dash/Streamlit  
- **Version Control & Collaboration:** Git, GitHub/GitLab  
- **Documentation & Reporting:** Markdown, LaTeX, Jupyter Notebooks  

---
