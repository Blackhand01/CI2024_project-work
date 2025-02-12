from datetime import datetime
from pathlib import Path
from utility.utils import initialize_experiment, load_data, run_genetic_programming, save_results
from core.statistics import GPStatistics
from utility.logger import Logger

def main():
    data_dir = './data'
    output_file = './s333971.py'
    base_output_dir = './experiments/'
    data_files = sorted(Path(data_dir).glob('*.npz'))

    if not data_files:
        print("No data files found in the directory.")
        return

    # Create a single global logger: this logger will write to a single file named general_log_<timestamp>.txt
    global_logger = Logger()

    for data_file in data_files:
        experiment_successful = True
        start_time = datetime.now()
        reason = "Max generations reached"
        stats = None
        best_individual = None

        try:
            # Initialize directories for the problem (without creating a new logger)
            experiment_config = initialize_experiment(data_file, base_output_dir)
            problem_id = experiment_config["problem_id"]
            plot_dir = experiment_config["plot_dir"]

            # Write a header in the log file for the current problem
            global_logger.log_problem_header(problem_id)
            global_logger.info(f"Processing Problem {problem_id}")

            # Load data
            x, y = load_data(data_file)

            # Run the GP algorithm and obtain results
            best_individual, stats = run_genetic_programming(x, y, global_logger)

            # Save the results (formula, plots, log)
            save_results(
                best_individual=best_individual,
                stats=stats,
                output_file=output_file,
                function_name=f"f{problem_id}",  # Function name based on the problem ID
                plot_dir=plot_dir,
            )

        except Exception as e:
            experiment_successful = False
            reason = f"Error: {str(e)}"
            global_logger.info(f"Error processing Problem {problem_id}: {reason}")
            print(f"Error processing Problem {problem_id}: {reason}")

        finally:
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()

            if stats is not None:
                best_expr = best_individual.tree_to_expression() if best_individual else "N/A"
                # Generate a summary for the current problem
                global_logger.generate_summary(
                    stats=stats,
                    best_expression=best_expr,
                    total_time=total_time,
                    start_time=start_time,
                    end_time=end_time,
                    reason=reason,
                    success=experiment_successful
                )
                # Store the summary for the final comparative table
                global_logger.store_problem_summary(
                    problem_id=problem_id,
                    best_fitness=stats.best_fitness,
                    diversity=stats.diversity,
                    generations_no_improv=stats.generations_no_improvement,
                    total_generations=stats.current_generation,
                    best_formula=best_expr,
                    total_time=total_time
                )
            else:
                global_logger.log_message("No GPStatistics available. An error may have occurred.")

    # At the end of all problems, print the global summary table
    global_logger.print_final_summary()
    print("All experiments completed.")

if __name__ == "__main__":
    main()
