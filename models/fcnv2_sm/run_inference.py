from inference_utils import run_custom_inference, run_simple_inference
import datetime
from time import perf_counter

# make sure to set the environment variable MODEL_REGISTRY before running this script
# see documentation in the README and the docstrings for both of these functions for more information
start = perf_counter()
run_custom_inference(
    data_source_path="/N/slate/jmelms/projects/FCN_dynamical_testing/data/initial_conditions/processed_ic_sets/default/",
    output_path="/N/slate/jmelms/projects/FCN_dynamical_testing/data/output/ideal_default_20t.nc",
    n_iters=20,
    start_time=datetime.datetime(1970, 1, 1),
    device="cuda", # cpu or cuda
    vocal=True
)

# run_simple_inference(
#     output_path="/N/slate/jmelms/projects/FCN_dynamical_testing/data/output/kwesi_test.nc",
#     n_iters=500,
#     start_time=datetime.datetime(2010, 1, 1),
#     device="cpu",
#     vocal=True
# )
stop = perf_counter()

print(f"Inference completed in: {stop-start:.2f} seconds")
