# Supplementary Document: Dataset Composition and Target Cohort Curation

### 1. Benchmark Origin and Characteristics
The ApacheJIT benchmark was compiled by Zeng et al. (ISSTA 2021) and deposited at Zenodo (DOI: 10.5281/zenodo.5907001). It extracts commit-level software changes, SZZ defect labels, and code metrics from 13 major open-source repositories managed by the Apache Software Foundation (ASF).

### 2. Benchmark Scale and Deduplication
- Total records in `apachejit_total.csv`: **106,674 rows**.
- Unique commit hashes (`commit_id`): **106,674**.
- Deduplication check confirms zero duplicate commits exist in the canonical benchmark file.

### 3. Repository Membership and Target Selection Criteria
We establish two explicit inclusion criteria for target evaluation:
1. **Scale Threshold:** Project must contain $\ge 2,000$ total commits.
2. **Defect Threshold:** Project's test period ($2016$–$2017$) must contain $\ge 50$ buggy commits.

Ten projects meet both criteria and form our primary target cohort:
`apache/camel`, `apache/hadoop`, `apache/ignite`, `apache/flink`, `apache/hbase`, `apache/cassandra`, `apache/groovy`, `apache/hive`, `apache/activemq`, `apache/kafka`.

Three repositories fail the 50-bug test threshold in 2016–2017 and serve exclusively as donor training sources:
- `apache/spark`: 1,465 commits (632 buggy; 4 test bugs in 2016–2017)
- `apache/zeppelin`: 1,451 commits (622 buggy; 18 test bugs in 2016–2017)
- `apache/zookeeper`: 839 commits (342 buggy; 6 test bugs in 2016–2017)

### 4. Merged Git Subprojects (Hadoop)
In early Apache history, Hadoop subprojects were hosted across separate repositories before being merged into the unified `apache/hadoop` root. In `apachejit_total.csv`, commits are recorded under:
- `apache/hadoop`: 11,964 commits
- `apache/hadoop-hdfs`: 2,907 commits
- `apache/hadoop-mapreduce`: 1,321 commits
Total unified Hadoop commits: **16,192**. All analyses merge these subproject tags into the canonical `apache/hadoop` project.
