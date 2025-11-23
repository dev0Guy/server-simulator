from server.envs.single_slot import SingleSlotCluster, SingleSlotClusterCreators


def random_machine_with_static_machine(n_machines: int, n_jobs: int, seed = None) -> SingleSlotCluster:
    return SingleSlotCluster(
        SingleSlotClusterCreators.random_workload_creator(n_jobs),
        SingleSlotClusterCreators.static_machine_creator(n_machines),
        seed=seed
    )


def static_machine_with_static_machine(n_machines: int, n_jobs: int, seed = None, static_value: float = 1.0) -> SingleSlotCluster:
    return SingleSlotCluster(
        SingleSlotClusterCreators.static_workload_creator(n_jobs, value=static_value),
        SingleSlotClusterCreators.static_machine_creator(n_machines),
        seed=seed
    )
