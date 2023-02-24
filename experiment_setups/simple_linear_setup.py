from benchmarks.mdp_tabular_direct_benchmark import MDPTabularDirectBenchmark
from benchmarks.mean_r_benchmark import MeanRBenchmark
from benchmarks.tennenholtz_benchmark import TennenholtzBenchmark
from benchmarks.time_independent_sampling_efficient_benchmark import \
    TimeIndependentSamplingEfficientBenchmark
from environments.pci_reducer import CurrPrevObsPCIReducer
from environments.simple_linear_environment import FixedActionVectorPolicy, \
    SimpleLinearEnvironment
from environments.toy_environment import ToyEnvironment, \
    ToyEvaluationPolicyEasy, ToyEvaluationPolicyHard, ToyEvaluationPolicyOptim
from methods.direct_pci_method import DirectPCIMethod
from methods.efficient_pci_method import EfficientPCIMethod
from methods.is_pci_method import ImportanceSamplingPCIMethod
from nuisance_estimation.discrete_single_kernel_nuisance_estimation import \
    DiscreteSingleKNuisanceEstimation
from utils.hyperparameter_optimization import HyperparameterPlaceholder
from utils.kernels import TripleMedianKernel
from utils.neural_nets import FlexibleCriticNet, TabularNet

psi_methods = [
    {
        "class": EfficientPCIMethod,
        "name": "EfficientKernelPCI",
        "args": {},
    },
    # {
    #     "class": ImportanceSamplingPCIMethod,
    #     "name": "ImportanceSamplingKernelPCI",
    #     "args": {},
    # },
    # {
    #     "class": DirectPCIMethod,
    #     "name": "DirectKernelPCI",
    #     "args": {},
    # },
]


nuisance_methods = [
    {
        "class": DiscreteSingleKNuisanceEstimation,
        "name": "SingleKernelNuisance",
        # "placeholder_options": {"alpha": [1e-2, 1e-4, 1e-6, 1e-8],
        #                         "lmbda": [1e0, 1e-2, 1e-4, 1e-6]},
        "placeholder_options": {},
        "num_folds": 5,
        "args": {
            "q_net_class": TabularNet,
            "q_net_args": {},
            "g_kernel_class": TripleMedianKernel,
            "g_kernel_args": {},
            "h_net_class": TabularNet,
            "h_net_args": {},
            "f_kernel_class": TripleMedianKernel,
            "f_kernel_args": {},
            # "q_alpha": HyperparameterPlaceholder("alpha"),
            # "h_alpha": HyperparameterPlaceholder("alpha"),
            # "q_lmbda": HyperparameterPlaceholder("lmbda"),
            # "h_lmbda": HyperparameterPlaceholder("lmbda"),
            "q_alpha": 1e-4,
            "h_alpha": 1e-4,
            "q_lmbda": 1e-6,
            "h_lmbda": 1e-6,
            "num_rep": 3,
        },
    },
]


benchmark_methods = [
    {
        "class": MDPTabularDirectBenchmark,
        "name": "MDPTabularDirectBenchmark",
        "args": {},
    },
    {
        "class": MeanRBenchmark,
        "name": "MeanRBenchmark",
        "args": {},
    },
]


n_range = [10000]
num_test = 20000
# num_test = 1000
horizon = 3
gamma = 1.0
num_reps = 100
num_procs = 1

pi_0 = {"class": FixedActionVectorPolicy,
        "args": {"action_vector": [1, 1, 1]}}

general_setup = {
    "pci_reducer": {
        "class": CurrPrevObsPCIReducer,
        "args": {},
    },
    "environment": {
        "class": SimpleLinearEnvironment,
        "args": {},
    },
    "n_range": n_range,
    "num_test": num_test,
    "horizon": horizon,
    "gamma": 1.0,
    "verbose": True,
    "num_reps": num_reps,
    "num_procs": num_procs,
    "psi_methods": psi_methods,
    "nuisance_methods": nuisance_methods,
    "benchmark_methods": benchmark_methods,
}

simple_linear_setup_0 = {"setup_name": "simple_linear_0",
                         "target_policy": pi_0, **general_setup}
