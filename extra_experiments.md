A. Experiments on Synthetic graphs
||||||SHD|SHD|SHD|Spurious Rate|Spurious Rate|Spurious Rate|Time|Time|Time|
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Scenario|#edges|#nodes|Varsort_score|R2sort_score|Varsort|R2sort|GLIDE|Varsort|R2sort|GLIDE|Varsort|R2sort|GLIDE|
|L-G|500|500|0.98|0.76|**92**|814|249.12|12.74|55.28|**1.98**|11.01|8.56|**27.25**|
|L-G|500|1000|0.98|0.92|739|4600|**539.6**|40.83|81.43|**5.78**|10.76|10.21|**15.19**|
|nL-nG|500|500|0.83|0.41|557|1054|**306.4**|47.75|61.16|**1.82**|10.24|10.25|**6.42**|
|nL-nG|500|1000|0.8|0.21|928|1531|**753.46**|39.41|48.35|**3.72**|9.52|7.98|**3.88**|

B. Experiments on Sachs dataset (2005):
| |SHD|Spurious Rate|Time|
|:---:|:---:|:---:|:---:|
|Varsort|33|0.64864865|0.28|
|R2sort|29|0.61538462|**0.16**|
|GLIDE|**8.7**|**0**|10.46|

C. Experiments on Real-world graphs:
| Insurance | Sortability | SHD | Spurious Rate | Time |
|:---:|:---:|:---:|:---:|:---:|
| Varsort | 0.64 | 105 | 62.01 | **2.54** |
| R2sort | 0.18 | 154 | 69.93 | 2.71 |
| GLIDE |   | **18** | **3.6** | 34.2 |

| Water | Sortability | SHD | Spurious Rate | Time |
|:---:|:---:|:---:|:---:|:---:|
| Varsort | 0.69 | 93 | 51.18 | 4.03 |
| R2sort | 0.41 | 145 | 62.96 | **3.39** |
| GLIDE |   | **41.6** | **23** | 49.1 |

| Alarm | Sortability | SHD | Spurious Rate | Time |
|:---:|:---:|:---:|:---:|:---:|
| Varsort | 0.55 | 108 | 65.87 | **5.29** |
| R2sort | 0.59 | 118 | 68.57 | 5.38 |
| GLIDE |   | **27.8** | **13.1** | 43.3 |

| Barley | Sortability | SHD | Spurious Rate | Time |
|:---:|:---:|:---:|:---:|:---:|
| Varsort | 0.76 | 127 | 54.33 | 7.78 |
| R2sort | 0.36 | 181 | 61.05 | **7.5** |
| GLIDE |   | **45.8** | **8.2** | 61.7 |

| Pathfinder | Sortability | SHD | Spurious Rate | Time |
|:---:|:---:|:---:|:---:|:---:|
| Varsort | 0.79 | 1496 | 89.29 | **43.91** |
| R2sort | 0.05 | 2433 | 92.94 | 44.94 |
| GLIDE |   | **59.1** | **1.9** | 197 |