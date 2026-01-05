Base Status
-----------
We've created 3 Environment for scheduling, which implement the `ClusterABC` with `JobCollection` and `MachineCollection` protocols: 


### SingleSlot Environment

| Component | Details |
|----------|---------|
| **Description** | Minimal scheduling environment with no temporal memory; each machine and job exposes only its current resource usage. |
| **State: Machines** | Shape: **[resource_usage]** — a single-cell array indicating current resource utilization (value between 0–1). |
| **State: Jobs** | Shape: **[resource_usage]** — a single-cell array indicating the current resource requirement (value between 0–1). |
| **Actions** | Discrete integer: **0 … (num_machines × num_jobs) + 1**. |
| **Action 0** | Interrupt / time tick (move to next timestamp). |
| **Action 1+** | Scheduling decision mapping to a specific **(machine, job)** pair. |

### DeepRM Environment

| Component | Details |
|----------|---------|
| **Description** | Environment inspired by the DeepRM scheduling model, representing fine-grained resource units across time. |
| **State: Machines** | Tensor shape: **[num_machines, num_resources, num_resource_cells, num_ticks]**. Each cell represents whether resource unit **r_unit** of resource **r** is occupied at time **t**. |
| **State: Jobs** | Tensor shape: **[num_jobs, num_resources, num_resource_cells, num_ticks]**. Each cell represents whether the job requires resource unit **r_unit** of resource **r** at time **t**. |
| **Actions** | Discrete integer: **0 … (num_machines × num_jobs) + 1**. |
| **Action 0** | Interrupt / time tick (advance to next timestamp). |
| **Action 1+** | Scheduling decision mapping to a specific **(machine, job)** pair. |

### MetricBased Environment

| Component | Details |
|----------|---------|
| **Description** | Scheduling environment that models resource usage of machines and jobs over time. |
| **State: Machines** | Tensor shape: **[num_machines, n_resources, n_ticks]**. Each value ∈ **[0–1]** representing machine resource usage for resource **r** at time **t**. |
| **State: Jobs** | Tensor shape: **[num_jobs, n_resources, n_ticks]**. Each value ∈ **[0–1]** representing job resource demand for resource **r** at time **t**. |
| **Actions** | Discrete integer: **0 … (num_machines × num_jobs) + 1**. |
| **Action 0** | Represents an interrupt or a time tick (advance to next timestamp). |
| **Action 1+** | Represents assigning job **j** to machine **m** (encoded according to environment mapping). |

<br>

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
- [-] Create Dilation Gym Environment Wrapper `DilationWrapper`.
- [-] Test `DilationWrapper`.
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
- [X] Test Dilation `MetricBasedDilator`. (need to improve tests)

## Assumption
- Dilation assume that cluster state is bigger than dilation & the kernel has no 1 in each of its diminution
- For each Step which is not real allocation reward is set to 0 
- On each job has reward of 1 if change status to running 
- Dilation is operating by taking [n_machine, n_resource, n_ticks] and
  padding to perpetrate size of [max_x_kernl, max_y_kernel, n_resources, n_ticks]
- Dilation implement both zoom in and zoom out when arriving to level 0 will cause real scheduling 
