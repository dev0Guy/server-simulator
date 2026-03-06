from src.envs.cluster_simulator.base.extractors.reward import DifferentInPendingJobsRewardCaculator
from src.envs.cluster_simulator.basic import BasicClusterEnv as BasicClusterEnv
from gymnasium import register

from src.envs.cluster_simulator.metric_based.creator import MetricBasedEnvCreator, MetricBasedCreatorParameters
from src.envs.cluster_simulator.single_slot.creator import *
from src.envs.cluster_simulator.deep_rm.creator import *

register(
    "ClusterScheduling-single-slot-v1",
    SingleSlotEnvCreator(),
    kwargs=SingleSlotCreatorParameters( # type: ignore
        n_jobs=10,
        n_machines=2,
        reward_caculator=DifferentInPendingJobsRewardCaculator(),
        seed=None
    )
)


register(
    "ClusterScheduling-deeprm-v1",
    DeepRMEnvCreator(),
    kwargs=DeepRMCreatorParameters(  # type: ignore
        n_jobs=10,
        n_machines=2,
        n_resources=3,
        n_resources_unit=5,
        n_ticks=5,
        reward_caculator=DifferentInPendingJobsRewardCaculator(),
        seed=None
    )
)


register(
    "ClusterScheduling-metric-offline-v1",
    MetricBasedEnvCreator(),
    kwargs=MetricBasedCreatorParameters( #  type: ignore
        n_jobs=10,
        n_machines=2,
        n_resources=3,
        n_ticks=5,
        poisson_lambda=4,
        offline=True,
        reward_caculator=DifferentInPendingJobsRewardCaculator(),
        seed=None
    )
)

register(
    "ClusterScheduling-metric-online-v1",
    MetricBasedEnvCreator(),
    kwargs=MetricBasedCreatorParameters( #  type: ignore
        n_jobs=10,
        n_machines=2,
        n_resources=3,
        n_ticks=5,
        poisson_lambda=4,
        offline=False,
        reward_caculator=DifferentInPendingJobsRewardCaculator(),
        seed=None
    )
)

