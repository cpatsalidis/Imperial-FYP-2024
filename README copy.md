# ICAART
 
 This is a PhD project on multi-agent self-organising systems, using social influence and learning for collective decision making. 

run.ipynb: to run the file, change initial parameter to run
run_repeated.ipynb: repetition of main loop for many times to produce aggregate/average results and observe emerging patterns
 ***
 source files in Folder src:
 1. init_helpers.py: initialisers of network, datacollectors and variables
 2. processesCloud.py: different functions needed for generating jobs and variables related to the specific problem and calculation of the different kinds of noises
 3. updates.py: update of attention of the agents to different voices, different functions are called based on the initial parameters (see inside thoryvos2 file for the details of the selection)
 4. thoryvos.py: main part of the code (for the RL regulator)
 Units folder contains function for the multi-agent system (model and agent initialisers)
 ***
 Figures folder contains figures created in run.ipynb
 csv folder contains the csv with the data from datacollectors
