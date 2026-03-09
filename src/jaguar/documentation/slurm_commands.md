

## Entry
ssh hpi-hpc

## Monitoring
- squeue --me
- tail -n 100 -f /sc/home/michele.marschner/project/jaguar_project/logs/array_[job_id]_[number].out

## Logs
less /sc/home/michele.marschner/project/jaguar_project/logs/array_<JOBID>_<TASKID>.out

## Launching

find configs/_generated/kaggle_deduplication -name "*.toml" | sort > generated_runs.txt
wc -l generated_runs.txt
sbatch slurm/your_array_script.slurm

## Prüfen
wc -l generated_runs.txt
cat generated_runs.txt
head -n 5 generated_runs.txt


## Cancel 
scancel [job_id]


## Git

git fetch --all --prune
git pull --no-rebase
git reset --hard origin/$(git branch --show-current)
git clean -fd
git branch --show-current
git branch -vv
git worktree list
git ls-tree -r origin/cluster --name-only | grep "experiments/round_2/splits"
git checkout origin/cluster -- experiments/round_2/splits


## Transfer
rsync -avz --progress michele.marschner@lx01.hpc.sci.hpi.de:/sc/home/michele.marschner/project/jaguar_project/experiments/round_2/kaggle_deduplication/ ~/Downloads/kaggle_deduplication/
