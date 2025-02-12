from datetime import datetime
from pathlib import Path
import os
import numpy as np
import importlib.util

from utility.utils import initialize_experiment, load_data, run_genetic_programming, save_results
from core.statistics import GPStatistics
from utility.logger import Logger

def get_prediction_for_problem(module_path: str, problem_id: str, x: np.ndarray) -> np.ndarray:
    """
    Dynamically imports the specified module (e.g., s333971.py) and extracts the function f{problem_id}.
    Executes the function on x, handling overflow and invalid values, and returns the predictions.
    """
    module_name = os.path.splitext(os.path.basename(module_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    func_name = f"f{problem_id}"
    if hasattr(module, func_name):
        func = getattr(module, func_name)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            y_pred = func(x)
            # Limit any NaN, +Inf, and -Inf values
            y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e6, neginf=-1e6)
        return y_pred
    else:
        raise ValueError(f"Function {func_name} not found in module {module_path}")

def main():
    data_dir = './data'
    output_file = './s333971.py'
    base_output_dir = './experiments/'
    data_files = sorted(Path(data_dir).glob('*.npz'))

    if not data_files:
        print("No data files found in the directory.")
        return

    # Create a single global logger
    global_logger = Logger()

    # for data_file in data_files:
    #     experiment_successful = True
    #     start_time = datetime.now()
    #     reason = "Max generations reached"
    #     stats = None
    #     best_individual = None

    #     try:
    #         experiment_config = initialize_experiment(data_file, base_output_dir)
    #         problem_id = experiment_config["problem_id"]
    #         plot_dir = experiment_config["plot_dir"]

    #         global_logger.log_problem_header(problem_id)
    #         global_logger.info(f"Processing Problem {problem_id}")

    #         x, y = load_data(data_file)

    #         best_individual, stats = run_genetic_programming(x, y, global_logger)

    #         save_results(
    #             best_individual=best_individual,
    #             stats=stats,
    #             output_file=output_file,
    #             function_name=f"f{problem_id}",
    #             plot_dir=plot_dir,
    #         )

    #     except Exception as e:
    #         experiment_successful = False
    #         reason = f"Error: {str(e)}"
    #         global_logger.info(f"Error processing Problem {problem_id}: {reason}")
    #         print(f"Error processing Problem {problem_id}: {reason}")

    #     finally:
    #         end_time = datetime.now()
    #         total_time = (end_time - start_time).total_seconds()

    #         if stats is not None:
    #             best_expr = best_individual.tree_to_expression() if best_individual else "N/A"
    #             global_logger.generate_summary(
    #                 stats=stats,
    #                 best_expression=best_expr,
    #                 total_time=total_time,
    #                 start_time=start_time,
    #                 end_time=end_time,
    #                 reason=reason,
    #                 success=experiment_successful
    #             )
    #             global_logger.store_problem_summary(
    #                 problem_id=problem_id,
    #                 best_fitness=stats.best_fitness,
    #                 diversity=stats.diversity,
    #                 generations_no_improvement=stats.generations_no_improvement,
    #                 total_generations=stats.current_generation,
    #                 best_formula=best_expr,
    #                 total_time=total_time
    #             )
    #         else:
    #             global_logger.log_message("No GPStatistics available. Possibly an error occurred.")

    # Now, for each dataset, evaluate the corresponding function (f0 for problem_0, f1 for problem_1, etc.)
    print("\nEvaluating functions in file s333971.py on the datasets:")
    for data_file in data_files:
        print(f"\nDataset: {data_file.name}")
        # Let's assume the file name is in the format "problem_X.npz"
        basename = data_file.stem  # e.g., "problem_0"
        parts = basename.split('_')
        if len(parts) < 2:
            print(f"  File name {data_file.name} does not conform to the expected format, skipping...")
            continue
        problem_id = parts[1]
        x, y = load_data(data_file)
        try:
            y_pred = get_prediction_for_problem(output_file, problem_id, x)
            mse = np.mean((y - y_pred) ** 2)
            # Calculate summary statistics for the predictions and the actual data
            stats_pred = {
                'mean': np.mean(y_pred),
                'std': np.std(y_pred),
                'min': np.min(y_pred),
                'max': np.max(y_pred)
            }
            stats_y = {
                'mean': np.mean(y),
                'std': np.std(y),
                'min': np.min(y),
                'max': np.max(y)
            }
            print(f"  Function f{problem_id}: MSE = {mse:.6f}")
            print(f"    Predictions: mean = {stats_pred['mean']:.6f}, std = {stats_pred['std']:.6f}, min = {stats_pred['min']:.6f}, max = {stats_pred['max']:.6f}")
            print(f"    Ground truth: mean = {stats_y['mean']:.6f}, std = {stats_y['std']:.6f}, min = {stats_y['min']:.6f}, max = {stats_y['max']:.6f}")
        except Exception as e:
            print(f"  Error evaluating function f{problem_id}: {e}")

    # Print the global summary table (if available)
    global_logger.print_final_summary()
    print("All experiments completed.")

if __name__ == "__main__":
    main()
