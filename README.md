# Learning and Deliberation for Requisite Social Influence

This repository gathers all the tools required to obtain the results outlined in the paper, which details my work on the Undergraduate Final Year Project. The project was built upon the preexisting notions and code presented in the `Requisite Social Influence in Self-Regulated Systems’ by Mrs. Mertzani.

## In the 'main' branch
 
Here we implement the second order cybernetics modification, that introduces the observer to the system. The key configurations and add-ons can be found in 'attendNoiseFBIEandR', 'update_historical_data' and 'find_past_selection' functions (/src/Helpers/processesCloud.py), 'workDynB' and 'rewardDyn' functions (/src/Helpers/thoryvos2.py) and 'updAttNInd' function (/src/Helpers/updates.py). Run 'run.ipynb' (1 run) and 'run_repeated.ipynb' (10 runs) files to obtain the desired results, by setting the update and reward parameters to 'ind'. Here we also have the A2C modification using stablebaselines3 (/src/Units/modelCloud.py), which can be run with update/reward parameters set to 'exp' and 'com'.

## In the 'partial' branch

Here we implement the partial observability with belief update algorithm modification. The key configurations and add-ons can be found in 'initialize_priors, 'update_beliefs' and 'compute_epoch_statistics' functions (/src/Helpers/processesCloud.py), 'reset_mamodel' and 'workDynB' functions (/src/Helpers/thoryvos2.py). To change the noise variance, modify the 'self.noise_variance' parameter in the 'initVarCloud' function (/src/Helpers/init_helpers.py)
