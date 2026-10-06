# Customer Clustering Results

## Baseline vs. Hybrid

The baseline uses standardized RFM features with K-Means++. The hybrid method filters DBSCAN noise before fitting K-Means. Hybrid metrics describe retained customers only.

- **transaction_rows**: 101
- **customer_count**: 30
- **cluster_count**: 3
- **silhouette**: 0.518108193990273
- **davies_bouldin**: 0.5740923057106437
- **calinski_harabasz**: 26.85812458331998
- **inertia**: 30.105462346041698
- **hybrid_silhouette**: 0.560828774353838
- **hybrid_davies_bouldin**: 0.4846527196708678
- **hybrid_retained_customers**: 16.0
- **hybrid_outliers**: 14.0
- **selected_eps**: 1.1274
- **selected_min_samples**: 2
- **selected_silhouette**: 0.388319537509405
- **selected_davies_bouldin**: 0.7872374730977357
- **selected_retention_pct**: 96.66666666666667
- **selected_silhouette_change_pct**: -25.05049292528746
- **selected_dbi_change_pct**: 37.12733392642302
- **smart_algorithm**: kmeans_flag_outliers
- **smart_reason**: small dataset (n=30) with 20.0% IQR outliers; preserve all customers
- **smart_retention**: 1.0
- **smart_outlier_count**: 6
- **smart_silhouette**: 0.5181
- **smart_davies_bouldin**: 0.5741
- **hybrid_silhouette_improvement_pct**: 8.245494060718737
- **hybrid_dbi_reduction_pct**: 15.579304085788534

## Segment Profiles

|   cluster |   customers |   avg_recency |   avg_frequency |   avg_monetary | profile      |
|----------:|------------:|--------------:|----------------:|---------------:|:-------------|
|         0 |           6 |       16.6667 |         7       |       16065.7  | Loyalists    |
|         1 |          23 |       39.9565 |         2.52174 |        1539.05 | Intermittent |
|         2 |           1 |      218      |         1       |          36    | At-Risk      |

## Automated Algorithm Selection

The selector profiles the standardized RFM data before choosing a pipeline.

|   n_samples |   n_features |   outlier_ratio |   mean_skew |   density_cv |   hopkins |
|------------:|-------------:|----------------:|------------:|-------------:|----------:|
|          30 |            3 |             0.2 |      2.0778 |       0.8004 |    0.7918 |

**Recommended algorithm:** `kmeans_flag_outliers`. **Reason:** small dataset (n=30) with 20.0% IQR outliers; preserve all customers.

The selected pipeline retains 100.0% of customers and flags 6 potential outliers without excluding them from K-Means assignment.

## DBSCAN Parameter Sensitivity Analysis

The sweep evaluates k-distance-derived epsilon values. The selected configuration requires at least 80% customer retention.

|    eps |   min_samples |   noise_pct |   clusters |   retention_pct |
|-------:|--------------:|------------:|-----------:|----------------:|
| 0.3459 |             2 |    46.6667  |          6 |         53.3333 |
| 0.5613 |             2 |    26.6667  |          2 |         73.3333 |
| 1.0418 |             2 |    10       |          2 |         90      |
| 1.1274 |             2 |     3.33333 |          3 |         96.6667 |
| 0.4954 |             3 |    36.6667  |          2 |         63.3333 |
| 0.7977 |             3 |    16.6667  |          1 |         83.3333 |
| 1.5043 |             3 |    10       |          1 |         90      |
| 2.1851 |             3 |     3.33333 |          1 |         96.6667 |
| 0.5381 |             4 |    43.3333  |          1 |         56.6667 |
| 0.8843 |             4 |    20       |          1 |         80      |
| 1.7532 |             4 |     3.33333 |          1 |         96.6667 |
| 2.3807 |             4 |     3.33333 |          1 |         96.6667 |
| 0.6384 |             5 |    33.3333  |          1 |         66.6667 |
| 1.0124 |             5 |    20       |          1 |         80      |
| 1.8233 |             5 |     6.66667 |          1 |         93.3333 |
| 2.6387 |             5 |     3.33333 |          1 |         96.6667 |

**Selected configuration:** eps=1.1274, min_samples=2, retention=96.7%, Silhouette=0.3883, DBI=0.7872.
Relative to the baseline, this retention-first setting changes Silhouette by -25.05% and DBI by 37.13%.

## Business Viability

The untuned hybrid result is useful for analysis but excluded too many customers. The selected configuration prioritizes broad customer coverage before optimizing quality metrics. On this small dataset, that trade-off retains customers but lowers cluster quality, so it should be treated as a coverage comparison rather than a replacement for the baseline.

## SDG Alignment

- **SDG 8:** Prioritize high-value and loyal customer segments for efficient growth.
- **SDG 9:** Apply reproducible analytics and hybrid machine-learning methods.
- **SDG 12:** Reduce marketing waste through targeted segment campaigns.
