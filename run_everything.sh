module add mpi/openmpi4-x86_64
mpic++ main.cpp -o program -fopenmp -std=c++11
python3 run_all_tasks.py

echo "Tasks have been sent. Wait for them to complete, then open the plot.ipynb notebook and run all cells."
