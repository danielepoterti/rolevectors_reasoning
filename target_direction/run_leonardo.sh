#!/bin/bash

#SBATCH --account=IscrC_LLM-EVAL
#SBATCH --job-name=lm_eval_%a
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
#SBATCH --time=15:00:00
#SBATCH --array=0-28         # 29 roles, indices 0 through 28
#SBATCH --output=role_%A_%a.out

# Define the roles array.
# Note: Items with spaces (like "data analyst") must be enclosed in quotes.
roles=(
  "economic researcher"
  "economist"
  "financial analyst"
  "electronics technician"
  "data scientist"
  "electrical engineer"
  "software engineer"
  "web developer"
  "bailiff"
  "lawyer"
  "data analyst"
  "mathematician"
  "statistician"
  "nurse"
  "doctor"
  "physician"
  "dentist"
  "surgeon"
  "geneticist"
  "biologist"
  "physicist"
  "teacher"
  "chemist"
  "ecologist"
  "politician"
  "sheriff"
  "enthusiast"
  "partisan"
  "psychologist"
)

# Get current role from the array using SLURM_ARRAY_TASK_ID
ROLE="${roles[$SLURM_ARRAY_TASK_ID]}"

echo "$(date): Job started"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing role: $ROLE"
echo "Running on node: $(hostname)"

# Load necessary modules
echo "Loading modules..."
module load python/3.11.6--gcc--8.5.0
echo "Modules loaded successfully"

# Set environment variables
echo "Setting up HF_HOME..."
export HF_HOME=/leonardo_work/IscrC_LLM-EVAL/dpoterti
echo "HF_HOME set to: $HF_HOME"

# Change to work directory
echo "Changing to work directory..."
cd "$WORK/dpoterti/rolevectors" || { echo "Error: Failed to change directory"; exit 1; }
echo "Successfully changed to work directory: $(pwd)"

# Activate virtual environment
echo "Activating virtual environment..."
source env/bin/activate || { echo "Error: Failed to activate virtual environment"; exit 1; }
echo "Virtual environment activated successfully"

# Change to target directory
echo "Changing to target directory..."
cd target_direction/ || { echo "Error: Failed to change to target directory"; exit 1; }
echo "Successfully changed to target directory: $(pwd)"

# Run pipeline command
echo "Starting pipeline execution..."
echo "Running model: meta-llama/Llama-3.2-1B-Instruct with role: $ROLE"
srun python -m pipeline.run_pipeline --model_path meta-llama/Llama-3.2-1B-Instruct --role "$ROLE"

# Final status
if [ $? -eq 0 ]; then
    echo "$(date): Pipeline completed successfully"
else
    echo "$(date): Pipeline failed with error code $?"
fi

echo "$(date): Job finished"
