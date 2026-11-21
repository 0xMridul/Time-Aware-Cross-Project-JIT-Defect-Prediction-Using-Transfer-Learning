# Hardware and Software Configuration

### Runtime Environment
- **Operating System:** macOS (Apple Silicon aarch64) / Linux (Ubuntu 22.04 LTS tested compatible)
- **Python Version:** Python 3.12.x (CPython)
- **Virtual Environment:** `uv` / `pip` / `conda`
- **Linear Algebra Acceleration:** Accelerate / OpenBLAS

### Computational Requirements
- **Memory:** Minimum 16 GB RAM recommended (for pooled training matrix covariance computation).
- **Disk:** ~2 GB for raw repository data and experimental output CSVs.
- **Runtime:**
  - Primary Analysis (LR + RF across 10 projects, 10 seeds): ~12 minutes on modern 8-core CPU.
  - Protocol D (6 historical gaps $\times$ 2 variants): ~35 minutes.
  - Full sensitivity suite: ~1.5 hours.
