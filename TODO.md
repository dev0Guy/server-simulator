Base Status
-----------
We've created 3 Environment for scheduling, which implement the `ClusterABC` with `JobCollection` and `MachineCollection` protocols: 


- **SingleSlot**: <br>
    **description:** <br>
    **states:** <br>
        **machines:** [resource usage], simple one cell nd array that represent usage in current time (no memory, value between 0 to 1).<br>
        **jobs:** [resource usage], simple one cell nd array that represent usage in current time (no memory, value between 0 to 1).<br>
    **actions:** discreate integer between 0 to (number of machine) * (number of jobs) + 1. where action 0 is for interrupt or ticking (moving to next timestamp).<br>


- **DeepRM**: <br>
    **description:** <br>
    **states:** <br>
        **machines:** [number_of_machines, number_of_resource, number_of_resource_cell, number_of_ticks], where each value inside the metrics is for time `t` and resource `r` does resource unit number `r_unit` is in usage.<br>
        **jobs:** [number_of_jobs, number_of_resource, number_of_resource_cell, number_of_ticks], where each value inside the metrics is for time `t` and resource `r` does resource unit number `r_unit` is in usage.<br>
    **actions:** discreate integer between 0 to (number of machine) * (number of jobs) + 1. where action 0 is for interrupt or ticking (moving to next timestamp).<br>


- **MetricBased**: <br>
    **description:** <br>
    **states:** <br>
        **machines:** [number_of_machines, n_resources, n_ticks], where each value inside the metrics is for time `t` and resource `r` is in usage (value between 0-1). <br>
        **jobs:** [number_of_jobs, n_resources, n_ticks], where each value inside the metrics is for time `t` and resource `r` is in usage (value between 0-1). <br>
    **actions:** discreate integer between 0 to (number of machine) * (number of jobs) + 1. where action 0 is for interrupt or ticking (moving to next timestamp). <br>

**Dilation Operations:** assuming kernel size (in each dimension) is bigger than 1. By padding the input according to max zoom in possible the service can work for varying kernel sizes. 
In addition, assume that use (in our case the DRL agent) can execute 3 operation with zoomingIn (going up one level -1) or zoomingOut (going out one level +1) or skipping to next timestamp, 
notice that without executing real scheduling (when in last level and select a machine) can't stop and skip.
Our algorithm represent the state as ndarray of `shape` to `[kernel_x, kernel_y, *shape[1:]]`. <br>

**Reward Functions:** for now the only reward function is +1 for scheduling job and changing the job status from pending into running.


Change Log
-----
EMPTY

Tasks:
-----
- [-] Test Dilation `MetricBasedDilator`. (need to improve tests)
- [-] Create Dilation Gym Environment Wrapper `DilationWrapper`.
- [ ] Test `DilationWrapper`.
- [ ] Create Dilation for DeepRM
- [ ] Create Tests for DeepRM
- [ ] Implement Render technics to represent and visualize cluster result
- [ ] Implement different reward Wrapper:
  -  [ ] Need to decide which one ?? 
  
Finished Tasks:
-----
- [X] Implement global machine Protocol (`Machine`,`MachineCollection`).
- [X] Implement global job Protocol (`Job`,`JobCollection`).
- [X] Implement abstract cluster using these protocols, for Cross-functional (`ClusterABC`), where each subclass implement the creation of jobs and machines.
- [X] Test `ClusterABC` (property based).
- [X] Implement Single slot cluster using the abstract class (`SingleSlotCluster`).
- [X] Test `SingleSlotCluster` (property based).
- [X] Implement DeepRM cluster using the abstract class (`DeepRMCluster`).
- [X] Test `DeepRMCluster` (property based).
- [X] Implement MetricBased cluster using the abstract class (`MetricCluster`).
- [X] Test `MetricCluster` (property based).
- [X] Implement Gym Environment that get cluster as dependency `BasicClusterEnv`.
- [X] Test `BasicClusterEnv` (property based) using random scheduler (`RandomScheduler`).
- [X] Implement Dilation Protocol `DilationProtocol`.
- [X] Implement Dilation numpy functionality (`hierarchical_pooling`, `get_window_from_cell`, etc..) `array_operation.py`.
- [X] Test Dilation numpy functionality.
- [X] Implement Dilation Service for Metric based cluster `MetricBasedDilator`.

## Assumption
- Dilation assume that cluster state is bigger than dilation & the kernel has no 1 in each of its diminution
- For each Step which is not real allocation reward is set to 0 
- On each job has reward of 1 if change status to running 
- Dilation is operating by taking [n_machine, n_resource, n_ticks] and
  padding to perpetrate size of [max_x_kernl, max_y_kernel, n_resources, n_ticks]
- Dilation implement both zoom in and zoom out when arriving to level 0 will cause real scheduling 
