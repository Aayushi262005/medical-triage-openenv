try:
    from openenv import BaseGrader
except (ImportError, AttributeError):
    BaseGrader = object


class MedicalTriageGrader(BaseGrader):
    def grade(self, episode_results) -> float:
        if not episode_results:
            return 0.01

        total_cases = len(episode_results)

        correct = 0
        partial = 0
        critical_miss = 0
        over_triage = 0
        has_info = False

        for step in episode_results:
            info = getattr(step, "info", {}) or {}

            predicted = info.get("predicted_level")
            actual = info.get("correct_level")

            if predicted is not None and actual is not None:
                has_info = True

                if predicted == actual:
                    correct += 1
                elif abs(predicted - actual) == 1:
                    partial += 1

                if info.get("is_critical") and predicted > actual:
                    critical_miss += 1

                if predicted < actual:
                    over_triage += 1

        if has_info and total_cases > 0:
            acc_score = correct / total_cases
            partial_score = partial / total_cases
            critical_penalty = critical_miss / total_cases
            over_penalty = over_triage / total_cases

            final_score = (
                0.6 * acc_score +
                0.25 * partial_score -
                0.5 * critical_penalty -
                0.05 * over_penalty
            )
        else:
            rewards = [getattr(step, "reward", 0.01) for step in episode_results]
            avg_reward = sum(rewards) / len(rewards) if rewards else 0.01
            final_score = avg_reward if avg_reward > 0 else 0.01

        # STRICT RANGE ENFORCEMENT
        if final_score <= 0.0:
            final_score = 0.01
        elif final_score >= 1.0:
            final_score = 0.99

        return float(round(final_score, 4))
