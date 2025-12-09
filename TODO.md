- [X] Implement Metric Based Tests
- [X] Move gym dependency outside of cluster (the state definition)
- [X] Implement Test For Basic Env
- [X] Implement Array operations that mimic dilation
- [X] Tests Array operations 
- [-] Create Custom Dilator for Metric Server [FINISHED NEED TO WAIT FOR TEST TO PASS]
- [-] Create Test for Dilator [Implemented by causes some errors]
- [-] Create Dilation Wrapper
- [ ] Create Test for dilation Wrapper
- [ ] Create Dilation for DeepRM
- [ ] Create Tests for DeepRM
- [ ] Implement Render technics to represent and visualize cluster result
- [ ] Implement different reward Wrapper:
  -  [ ] Need To talk and decide on them 
----
## Notice:
- Dilation assume that cluster state is bigger than dilation & the kernel has no 1 in each of its diminution
- For each Step which is not real allocation reward is set to 0 
- On each job has reward of 1 if change status to running 
- Dilation is operating by taking [n_machine, n_resource, n_ticks] and
  padding to perpetrate size of [max_x_kernl, max_y_kernel, n_resources, n_ticks]
  where log(n) based of kernel_x will be max_x_kernl.
- Dilation implement both zoom in and zoom out when arriving to level 0 will cause real scheduling 