# MIT-BIH Arrhythmia support for medtsllm-biomedcoop

This patch adds MIT-BIH as a separate dataset and leaves PTB-XL intact.

## What the patch changes

- Adds `datasets/mitdb.py`
- Registers `MITDB` in `datasets/__init__.py`
- Adds AAMI five-class beat mapping: `N`, `S`, `V`, `F`, `Q`
- Uses a patient-independent DS1/DS2 split
- Extracts a fixed ECG window centered on each annotated beat
- Uses MLII when available
- Adds MIT-BIH BioMedCoOp prompt descriptions
- Adds optional square-root inverse-frequency weighted cross-entropy
- Adds `scripts/check_mitdb.py`
- Adds `wfdb==4.1.2` to `requirements.txt`

## 1. Apply the patch

Run from the repository root:

```bash
cd /lambda/nfs/Kiana5/medtsllm-biomedcoop

git status
git switch -c add-mitdb

git apply /path/to/medtsllm_mitdb_patch/mitdb_changes.patch
```

Check the changes:

```bash
git status
git diff --check
```

If `git apply` says a target file differs from the public branch, copy the new
files from this bundle and make the three small edits shown in
`mitdb_changes.patch` manually.

## 2. Activate the environment and install WFDB

```bash
source /lambda/nfs/Kiana5/medtsllm/bin/activate
cd /lambda/nfs/Kiana5/medtsllm-biomedcoop

python3 -m pip install "wfdb==4.1.2"
```

Installing the complete repository requirements is optional when the current
environment already runs PTB-XL:

```bash
python3 -m pip install -r requirements.txt
```

## 3A. Use the ZIP you already downloaded

Suppose the ZIP is at `/lambda/nfs/Kiana5/mitdb-1.0.0.zip`:

```bash
cd /lambda/nfs/Kiana5/medtsllm-biomedcoop
mkdir -p data/mitdb data/mitdb_extract
rm -rf data/mitdb_extract/*

unzip /lambda/nfs/Kiana5/mitdb-1.0.0.zip -d data/mitdb_extract

RECORDS_FILE="$(find data/mitdb_extract -type f -name RECORDS -print -quit)"
test -n "$RECORDS_FILE" || { echo "RECORDS was not found in the ZIP"; exit 1; }
MITDB_SRC="$(dirname "$RECORDS_FILE")"

cp -a "$MITDB_SRC"/. data/mitdb/
rm -rf data/mitdb_extract

ls -lh data/mitdb/100.hea data/mitdb/100.dat data/mitdb/100.atr
```

A `416 Requested Range Not Satisfiable` response from `wget -c` normally means
the local output file is already as large as the remote file. Validate the ZIP:

```bash
unzip -t /lambda/nfs/Kiana5/mitdb-1.0.0.zip
```

## 3B. Alternative: download through the WFDB package

Use this instead of step 3A:

```bash
cd /lambda/nfs/Kiana5/medtsllm-biomedcoop
mkdir -p data/mitdb

python3 - <<'PY'
from pathlib import Path
import wfdb

target = Path("data/mitdb")
target.mkdir(parents=True, exist_ok=True)
wfdb.dl_database("mitdb", dl_dir=str(target))
print("Downloaded MIT-BIH to", target.resolve())
PY
```

## 4. Validate preprocessing and build the cache

```bash
python3 scripts/check_mitdb.py
```

The first execution reads all WFDB records and creates compressed files under:

```text
data/mitdb/cache/
```

The printed tensors should have this form:

```text
train: samples=..., shape=(..., 512, 1), counts={'N': ..., 'S': ..., 'V': ..., 'F': ..., 'Q': ...}
  val: samples=..., shape=(..., 512, 1), counts={...}
 test: samples=..., shape=(..., 512, 1), counts={...}
```

## 5. Confirm access to the configured LLM

The supplied config follows the repository's PTB-XL decoder config and uses
`meta-llama/Llama-2-7b-hf`. It is gated on Hugging Face.

```bash
huggingface-cli login
```

If your working PTB-XL configuration already uses another accessible LLM,
replace only this line in `configs/datasets/mitdb_biomedcoop.toml`:

```toml
llm = "meta-llama/Llama-2-7b-hf"
```

with the same model ID that already works in your environment.

## 6. Run a smoke test

Temporarily reduce the config to one epoch:

```bash
cp configs/datasets/mitdb_biomedcoop.toml \
   configs/datasets/mitdb_biomedcoop_smoke.toml

sed -i 's/epochs = 20/epochs = 1/' \
  configs/datasets/mitdb_biomedcoop_smoke.toml
sed -i 's/batch_size = 8/batch_size = 2/' \
  configs/datasets/mitdb_biomedcoop_smoke.toml

python3 train.py configs/datasets/mitdb_biomedcoop_smoke.toml
```

## 7. Run the full experiment

```bash
python3 train.py configs/datasets/mitdb_biomedcoop.toml
```

If CUDA runs out of memory, change:

```toml
batch_size = 8
```

to `4`, `2`, or `1`. If bitsandbytes is installed and supported by the GPU, you
can also try:

```toml
load_in_4bit = true
```

## Important experimental details

- PTB-XL performs one label per complete ECG record.
- MIT-BIH provides one annotation per heartbeat.
- This implementation therefore performs heartbeat classification, not
  record-level diagnostic classification.
- DS1 contributes training and validation patients; DS2 remains the test set.
- The `Q` class is very small. Use macro-F1 in addition to accuracy and inspect
  the confusion matrix before drawing conclusions.
- `trim_beats = 10` discards ten mapped beats at each end of every record.
- `history_len = 512` corresponds to approximately 1.42 seconds at 360 Hz.
- To use 256 samples, set both `history_len` and `pred_len` to 256, then delete
  `data/mitdb/cache/` so the cache is rebuilt.
