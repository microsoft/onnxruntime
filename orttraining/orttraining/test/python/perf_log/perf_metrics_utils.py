from datetime import datetime
import git
from create_table_perf_test_training_ort_module_data import ConnectAndInsertPerfMetrics

def get_repo_commit(repo_path):
    repo = git.Repo(repo_path, search_parent_directories=True)
    sha = repo.head.object.hexsha
    short_sha = repo.git.rev_parse(sha, short=4)
    return short_sha

class PerfLog:
    def __init__(self):
        self.perf_metrics = {}
        self.number_of_batches_ = 0
        self.gradient_accumulation_step_count
        self.weight_update_steps
        self.total_time
        self.avg_time_per_batch
        self.throughput
        self.stabilized_throughput
        self.e2e_throughput
        self.mapped_dimensions
        self.average_cpu_usage
        self.peak_workingset_size

    def log_current_train_step(self):
        return

    def complete(self):
        self.compute_metrics()
        ConnectAndInsertPerfMetrics()

    def compute_metrics(self):
        self.perf_metrics['Model'] = "model"
        self.perf_metrics['BatchId'] = "batch_id2"
        self.perf_metrics['CommitId'] = "commit id"
        self.perf_metrics['ModelName'] = "model name"
        self.perf_metrics['DisplayName'] = "disp name"
        self.perf_metrics['UseMixedPrecision'] = True
        self.perf_metrics['UseAutoCast'] = False
        self.perf_metrics['UseDeepSpeed'] = True
        self.perf_metrics['Optimizer'] = "optim"
        self.perf_metrics['BatchSize'] = 20
        self.perf_metrics['SeqLen'] = 128
        self.perf_metrics['PredictionsPerSeq'] = 200
        self.perf_metrics['NumOfBatches'] = 300
        self.perf_metrics['WeightUpdateSteps'] = 20
        self.perf_metrics['Round'] = 10
        self.perf_metrics['GradAccSteps'] = 12
        self.perf_metrics['AvgTimePerBatch'] = 0.2
        self.perf_metrics['Throughput'] = 20
        self.perf_metrics['StabilizedThroughput'] = 30
        self.perf_metrics['EndToEndThroughput'] = 40
        self.perf_metrics['TotalTime'] = 2.3
        self.perf_metrics['AvgCPU'] = 2
        self.perf_metrics['Memory'] = 3
        self.perf_metrics['RunConfig'] ="run  config"
        self.perf_metrics['Time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

