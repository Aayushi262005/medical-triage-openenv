import openenv
BaseGrader = getattr(openenv, "BaseGrader", object)

class MedicalTriageGrader(BaseGrader):
    def grade(self, episode_results) -> float:
        """
        Calculates a normalized score between 0.0 and 1.0.
        Scores are based on the average reward per patient.
        """
        if not episode_results:
            return 0.0
            
        total_reward = sum(step.reward for step in episode_results)
        num_steps = len(episode_results)
        
        # Calculate average reward
        avg_reward = total_reward / num_steps
        
        final_score = max(0.0, min(1.0, avg_reward))
        
        return float(final_score)