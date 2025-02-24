#!/bin/bash

#SBATCH --account=IscrC_LLM-EVAL
#SBATCH --job-name=lm_eval_mathematician
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
#SBATCH --time=15:00:00
#SBATCH --output=role_mathematician_%j.out

echo "$(date): Job started"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Processing role: mathematician"

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
echo "Running model: deepseek-ai/DeepSeek-R1-Distill-Llama-8B with role: mathematician"
srun python -m pipeline.run_pipeline --model_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B --role "mathematician"

# Final status
if [ $? -eq 0 ]; then
    echo "$(date): Pipeline completed successfully"
else
    echo "$(date): Pipeline failed with error code $?"
fi

echo "$(date): Job finished"
