# inst414-actor-clusters
# Actor Clusters by Genre (INST414 Module 3)

This project has 3,199 IMDB actors and groups them(5+ films, 2000–2022)


## How to run
pip install pandas numpy scikit-learn matplotlib requests
python cluster_actors.py

The script downloads the data from the course GitHub archive.

## What it produces
- Inertia and silhouette scores for k = 2 to 20 (used to choose k)
- The genre information of each cluster and example actors in each
- fig_k_selection.png and fig_cluster_profiles.png
