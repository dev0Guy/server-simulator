from src.envs.deep_rm import DeepRMJobs


def get_index_of_min_job_arrival_time(jobs: DeepRMJobs):
    j_idx, _ = min(
        enumerate(iter(jobs)),
        key=lambda t: t[1].arrival_time
    )
    return j_idx

def get_index_of_max_job_arrival_time(jobs: DeepRMJobs):
    j_idx, _ = max(
        enumerate(iter(jobs)),
        key=lambda t: t[1].arrival_time
    )
    return j_idx