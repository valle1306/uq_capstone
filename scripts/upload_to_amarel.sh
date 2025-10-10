#!/bin/bash
# Script to upload BRATS data to Amarel cluster
# 
# Usage:
#   bash scripts/upload_to_amarel.sh
#
# Before running:
# 1. Update YOUR_NETID below with your actual Rutgers NetID
# 2. Make sure you've prepared the data locally first
# 3. Ensure you can SSH to Amarel (test with: ssh YOUR_NETID@amarel.rutgers.edu)

# ========== CONFIGURATION ==========
YOUR_NETID="YOUR_NETID_HERE"  # <<< CHANGE THIS!
AMAREL_HOST="amarel.rutgers.edu"
AMAREL_USER="${YOUR_NETID}"

# Local paths (adjust if needed)
LOCAL_DATA_DIR="data/brats"
LOCAL_SCRIPTS_DIR="scripts"
LOCAL_SRC_DIR="src"
LOCAL_ENVS_DIR="envs"
LOCAL_NOTEBOOKS_DIR="notebooks"

# Remote paths on Amarel
AMAREL_SCRATCH="/scratch/${AMAREL_USER}"
AMAREL_PROJECT_DIR="${AMAREL_SCRATCH}/uq_capstone"
AMAREL_DATA_DIR="${AMAREL_PROJECT_DIR}/data/brats"

# ===================================

echo "========================================"
echo "  Uploading UQ Capstone to Amarel"
echo "========================================"
echo "Target: ${AMAREL_USER}@${AMAREL_HOST}"
echo "Remote project dir: ${AMAREL_PROJECT_DIR}"
echo "========================================"
echo ""

# Check if local data directory exists
if [ ! -d "$LOCAL_DATA_DIR" ]; then
    echo "ERROR: Local data directory not found: $LOCAL_DATA_DIR"
    echo "Please run data preparation scripts first!"
    exit 1
fi

# Check if NetID was changed
if [ "$YOUR_NETID" = "YOUR_NETID_HERE" ]; then
    echo "ERROR: Please edit this script and set YOUR_NETID!"
    exit 1
fi

echo "Step 1: Creating remote directories..."
ssh ${AMAREL_USER}@${AMAREL_HOST} << EOF
    mkdir -p ${AMAREL_PROJECT_DIR}/{data/brats,scripts,src,envs,notebooks,runs}
    echo "Remote directories created successfully"
EOF

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create remote directories. Check your SSH connection."
    exit 1
fi

echo ""
echo "Step 2: Uploading data files..."
echo "  (This may take a while depending on data size)"
rsync -avz --progress \
    ${LOCAL_DATA_DIR}/ \
    ${AMAREL_USER}@${AMAREL_HOST}:${AMAREL_DATA_DIR}/

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to upload data files"
    exit 1
fi

echo ""
echo "Step 3: Uploading scripts..."
rsync -avz --progress \
    ${LOCAL_SCRIPTS_DIR}/*.py \
    ${LOCAL_SCRIPTS_DIR}/*.sbatch \
    ${AMAREL_USER}@${AMAREL_HOST}:${AMAREL_PROJECT_DIR}/scripts/

echo ""
echo "Step 4: Uploading source code..."
rsync -avz --progress \
    ${LOCAL_SRC_DIR}/*.py \
    ${AMAREL_USER}@${AMAREL_HOST}:${AMAREL_PROJECT_DIR}/src/

echo ""
echo "Step 5: Uploading environment files..."
rsync -avz --progress \
    ${LOCAL_ENVS_DIR}/*.yml \
    requirements.txt \
    ${AMAREL_USER}@${AMAREL_HOST}:${AMAREL_PROJECT_DIR}/

echo ""
echo "Step 6: Uploading notebooks..."
rsync -avz --progress \
    ${LOCAL_NOTEBOOKS_DIR}/*.ipynb \
    ${AMAREL_USER}@${AMAREL_HOST}:${AMAREL_PROJECT_DIR}/notebooks/

echo ""
echo "========================================"
echo "✓ Upload complete!"
echo "========================================"
echo ""
echo "Next steps on Amarel:"
echo "  1. SSH to Amarel:"
echo "     ssh ${AMAREL_USER}@${AMAREL_HOST}"
echo ""
echo "  2. Navigate to project directory:"
echo "     cd ${AMAREL_PROJECT_DIR}"
echo ""
echo "  3. Set up conda environment:"
echo "     module load conda"
echo "     conda env create -f envs/conda_env.yml"
echo "     conda activate uq_capstone"
echo ""
echo "  4. Verify data:"
echo "     python scripts/validate_brats_data.py --data_root data/brats"
echo ""
echo "  5. Submit a test job:"
echo "     sbatch scripts/train_single.sbatch"
echo ""
echo "========================================"
