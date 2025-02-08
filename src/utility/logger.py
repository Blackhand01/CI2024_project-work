import logging
import os
import csv
from datetime import datetime
from memetic_config import POP_SIZE, MAX_DEPTH, N_GENERATIONS, TOURNAMENT_SIZE, MUTATION_RATE, CROSSOVER_RATE, ELITISM, BLOAT_PENALTY

class Logger:
    # File globale in cui verrà salvato il riepilogo finale di tutti gli esperimenti
    GLOBAL_SUMMARY_FILE = os.path.join("./docs", "global_evolution_summary.txt")

    def __init__(self, log_dir="./logs", log_file_prefix="gp_run"):
        # Crea la cartella docs se non esiste
        if not os.path.exists("./docs"):
            os.makedirs("./docs")

        # Crea la cartella log_dir se non esiste
        os.makedirs(log_dir, exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # File CSV contenente metriche e messaggi per questo specifico esperimento
        self.log_file = os.path.join(log_dir, f"{log_file_prefix}_{self.timestamp}_log.csv")

        # File di log testuale generale (uno per ogni esecuzione)
        self.general_log_file = os.path.join("./docs", f"general_log_{self.timestamp}.txt")

        # Header per la tabella delle metriche
        self.metrics_fields = [
            "timestamp",
            "generation",
            "best_fitness",
            "average_fitness",
            "diversity",
            "complexity",
            "selection_strategy",
            "crossover_strategy",
            "mutation_strategy",
            "local_search_algorithm"
        ]
        # Header per la tabella dei messaggi
        self.messages_fields = [
            "timestamp",
            "message"
        ]
        # Liste interne per accumulare log di metriche e messaggi
        self.metrics_logs = []
        self.messages_logs = []

        # Lista per memorizzare i summary dei vari problemi processati
        self.problems_summary = []

        # Configurazione del logging (console + file generale)
        logging.basicConfig(
            format="%(asctime)s - %(message)s",
            level=logging.INFO,
            handlers=[
                logging.StreamHandler(),  # Output su console
                logging.FileHandler(self.general_log_file, mode='a')  # Output su file generale
            ]
        )
        self.logger = logging.getLogger("GPLogger")

    def log_metrics(self, generation=None, best_fitness=None, avg_fitness=None,
                    diversity=None, complexity=None, strategies=None, local_search=None):
        """Accumula una riga di metriche nella lista interna."""
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "generation": generation if generation is not None else "",
            "best_fitness": f"{best_fitness:.4f}" if best_fitness is not None else "",
            "average_fitness": f"{avg_fitness:.4f}" if avg_fitness is not None else "",
            "diversity": f"{diversity:.4f}" if diversity is not None else "",
            "complexity": f"{complexity:.4f}" if complexity is not None else "",
            "selection_strategy": strategies.get("selection") if strategies else "",
            "crossover_strategy": strategies.get("crossover") if strategies else "",
            "mutation_strategy": strategies.get("mutation") if strategies else "",
            "local_search_algorithm": local_search if local_search else ""
        }
        self.metrics_logs.append(row)

    def log_message(self, message):
        """
        Accumula un messaggio (o una lista di messaggi) nella lista interna dei messaggi.
        """
        if isinstance(message, list):
            message = " | ".join(message)
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "message": message
        }
        self.messages_logs.append(row)

    def info(self, message, generation=None, best_fitness=None, avg_fitness=None,
             diversity=None, complexity=None, strategies=None, local_search=None):
        """
        Logga un messaggio informativo su console/file e, se presenti, salva anche le metriche.
        """
        if isinstance(message, list):
            message = " | ".join(message)
        # Log su console e file generale
        self.logger.info(message)
        # Se ci sono metriche, aggiunge una riga alla sezione metriche
        if any(v is not None for v in [
            generation, best_fitness, avg_fitness,
            diversity, complexity, strategies, local_search
        ]):
            self.log_metrics(
                generation, best_fitness, avg_fitness,
                diversity, complexity, strategies, local_search
            )
        # Logga sempre il messaggio nella sezione dei messaggi
        self.log_message(message)

    def interpret_metric(self, value, thresholds):
        """
        Restituisce una stringa descrittiva ('Low', 'Medium' o 'High')
        in base ai valori di soglia specificati.
        """
        if value <= thresholds[0]:
            return "Low"
        elif value <= thresholds[1]:
            return "Medium"
        else:
            return "High"

    def generate_summary(self, stats, best_expression, total_time, start_time, end_time, reason, success=True):
        """
        Genera un sommario dettagliato e lo aggiunge sia ai log locali sia al file globale.
        """
        strategy_usage = stats.get_strategy_usage()
        diversity_category = self.interpret_metric(stats.diversity, [0.5, 2])
        complexity_category = self.interpret_metric(stats.complexity, [5, 10])
        fitness_category = (
            "Best" if stats.best_fitness == 0 else
            "Good" if stats.best_fitness < 0.5 else
            "Discrete" if stats.best_fitness < 1 else
            "Bad"
        )
        stagnation_percentage = (
            0 if stats.current_generation == 0
            else (stats.generations_no_improvement / stats.current_generation) * 100
        )

        summary = (
            "\n==================== Experiment Summary ====================\n"
            "--- Experiment Configuration ---\n"
            f"Population Size: {POP_SIZE}\n"
            f"Max Depth: {MAX_DEPTH}\n"
            f"Max Generations: {N_GENERATIONS}\n"
            f"Mutation Rate: {MUTATION_RATE}\n"
            f"Crossover Rate: {CROSSOVER_RATE}\n"
            f"Elitism: {ELITISM}\n\n"
            "--- Experiment Status ---\n"
            f"Experiment started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Experiment finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Duration: {total_time:.2f} seconds\n"
            f"Status: {'Succeeded' if success else 'Failed'}\n"
            f"Reason: {reason}\n\n"
            "--- Experiment Statistics ---\n"
            f"Final Diversity: {stats.diversity:.4f} ({diversity_category})\n"
            f"Final Complexity: {stats.complexity:.4f} ({complexity_category})\n"
            f"Total Generations: {stats.current_generation}\n"
            f"Generations Without Improvement: {stats.generations_no_improvement} "
            f"({stagnation_percentage:.1f}%)\n"
            f"Best Fitness Achieved: {stats.best_fitness:.4f} ({fitness_category})\n"
            f"Best Expression: {best_expression}\n\n"
            "--- Strategy Usage ---\n"
            f"Selection Strategies: {strategy_usage['selection']}\n"
            f"Crossover Strategies: {strategy_usage['crossover']}\n"
            f"Mutation Strategies: {strategy_usage['mutation']}\n"
            f"Local Search Algorithms: {strategy_usage.get('local_search', {})}\n"
        )

        # Logga il sommario nella console e nei messaggi locali
        self.logger.info(summary)
        self.log_message(summary)

        # Aggiunge il sommario al file globale di riepilogo
        with open(self.GLOBAL_SUMMARY_FILE, "a") as f:
            f.write(summary)
            f.write("\n" + "="*60 + "\n")

        # Salva su CSV
        self.flush_logs()

    def flush_logs(self):
        """
        Salva metriche e messaggi in un unico file CSV, con due sezioni separate.
        """
        with open(self.log_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # Sezione metriche
            writer.writerow(["# Metrics"])
            writer.writerow(self.metrics_fields)
            for row in self.metrics_logs:
                writer.writerow([row[field] for field in self.metrics_fields])
            writer.writerow([])  # Riga vuota per separare
            # Sezione messaggi
            writer.writerow(["# Algorithm strategies track"])
            writer.writerow(self.messages_fields)
            for row in self.messages_logs:
                writer.writerow([row[field] for field in self.messages_fields])

    def log_problem_header(self, problem_id):
        """
        Logga un header per separare i log di un nuovo problema.
        """
        separator = "-" * 40
        header = f"\n{separator}\nProblem {problem_id}\n{separator}\n"
        self.logger.info(header)
        self.log_message(header)

    def store_problem_summary(self, problem_id, best_fitness, diversity,
                              generations_no_improv, total_generations,
                              best_formula, total_time):
        """
        Salva le metriche essenziali per il problema corrente in una lista per il riepilogo finale.
        """
        self.problems_summary.append({
            "problem_id": problem_id,
            "best_fitness": best_fitness,
            "diversity": diversity,
            "gen_no_improv": generations_no_improv,
            "total_gens": total_generations,
            "time_sec": total_time,
            "best_formula": best_formula
        })

    def print_final_summary(self):
        """
        Stampa e logga una tabella riassuntiva dei problemi processati.
        """
        summary_lines = []
        summary_lines.append("\n********** GLOBAL SUMMARY **********\n")
        header = f"{'PROB':<6} {'BEST_FIT':>10} {'DIVERSITY':>10} {'GEN_NO_IMP':>12} {'TOTAL_GEN':>10} {'TIME(s)':>10}   {'BEST_FORMULA':<40}"
        summary_lines.append(header)
        summary_lines.append("-" * len(header))
        for s in self.problems_summary:
            best_formula = s["best_formula"]
            if len(best_formula) > 37:
                best_formula = best_formula[:37] + "..."
            line = f"{s['problem_id']:<6} {s['best_fitness']:>10.4f} {s['diversity']:>10.4f} {s['gen_no_improv']:>12} {s['total_gens']:>10} {s['time_sec']:>10.2f}   {best_formula:<40}"
            summary_lines.append(line)
        summary_lines.append("\n" + "*" * 60 + "\n")
        final_summary = "\n".join(summary_lines)
        self.logger.info(final_summary)
        self.log_message(final_summary)
        # Aggiunge il riepilogo al file globale di riepilogo
        with open(self.GLOBAL_SUMMARY_FILE, "a") as f:
            f.write(final_summary)
            f.write("\n" + "*" * 60 + "\n")
