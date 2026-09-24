export interface Task {
  id: number;
  title: string;
  description?: string;
  completed: boolean;
  created_at: string;
  priority?: string;
  prediction_probability?: number;
  risk_level?: "low" | "medium" | "high";
  recommended_action?: "monitor" | "follow_up" | "escalate";
  intervention_required?: boolean;
  suggested_priority?: string;
  prediction_reason?: string;
}
