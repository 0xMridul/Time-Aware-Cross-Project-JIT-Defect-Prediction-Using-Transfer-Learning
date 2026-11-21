# Data Dictionary: ApacheJIT Schema and Attribute Definitions

The canonical dataset `apachejit_total.csv` contains the following schema:

| Column Name | Data Type | Units / Range | Description |
|:---|:---:|:---:|:---|
| `commit_id` | String | 40-char SHA-1 | Unique Git commit identifier |
| `project` | String | Categorical | Apache repository path (e.g., `apache/camel`) |
| `buggy` | Boolean | True / False | SZZ defect classification (1 = bug-inducing, 0 = clean) |
| `fix` | Boolean | True / False | Commit purpose indicator (1 = issue fix, 0 = other; mined retrospectively from Jira resolution status, excluded from primary evaluation to prevent prediction-time leakage) |
| `year` | Integer | 2004–2019 | Calendar year of the commit author date |
| `author_date` | Integer | Unix Timestamp (s) | Author timestamp in seconds since UTC epoch |
| `la` | Float | $\ge 0$ | Lines of code added |
| `ld` | Float | $\ge 0$ | Lines of code deleted |
| `nf` | Float | $\ge 1$ | Number of modified files |
| `nd` | Float | $\ge 1$ | Number of modified directories |
| `ns` | Float | $\ge 1$ | Number of modified subsystems |
| `ent` | Float | $[0, 1]$ | Shannon entropy across modified files |
| `ndev` | Float | $\ge 0$ | Number of prior developers modifying the files |
| `age` | Float | Days ($\ge 0$) | Average age of modified files since prior change |
| `nuc` | Float | $\ge 0$ | Number of unique prior changes |
| `aexp` | Float | $\ge 0$ | Geometric author developer experience |
| `arexp` | Float | $\ge 0$ | Recent author developer experience |
| `asexp` | Float | $\ge 0$ | Author experience in specific subsystems |
