import unittest

from backend_fastapi.ops import (
    calculate_llm_cost,
    estimate_tokens,
    evaluate_retrieval,
    evaluate_answer_quality,
    embedding_drift,
    embedding_model_manifest,
    get_metrics_snapshot,
    get_runtime_metadata,
    redact_payload,
    record_llm_observation,
    reset_metrics,
)


class OpsMetricsTests(unittest.TestCase):
    def tearDown(self):
        reset_metrics()

    def test_retrieval_metrics_reward_high_rank_hits(self):
        metrics = evaluate_retrieval(["task-2", "task-1", "task-3"], ["task-1"], k=3)

        self.assertEqual(metrics["precision_at_k"], 1 / 3)
        self.assertEqual(metrics["recall_at_k"], 1.0)
        self.assertEqual(metrics["mrr_at_k"], 1 / 2)
        self.assertEqual(metrics["ndcg_at_k"], 1 / 1.584962500721156)

    def test_retrieval_metrics_handle_empty_results(self):
        self.assertEqual(
            evaluate_retrieval([], [1]),
            {
                "precision_at_k": 0.0,
                "recall_at_k": 0.0,
                "mrr_at_k": 0.0,
                "ndcg_at_k": 0.0,
            },
        )

    def test_token_estimate_is_deterministic(self):
        self.assertEqual(estimate_tokens("12345678"), 2)
        self.assertEqual(estimate_tokens("123456789"), 3)

    def test_runtime_metadata_has_operational_versions(self):
        metadata = get_runtime_metadata()

        self.assertIn("service_version", metadata)
        self.assertIn("embedding_model", metadata)
        self.assertIn("prompt_version", metadata)
        self.assertIn("llm_model", metadata)

    def test_sensitive_values_are_redacted_recursively(self):
        payload = redact_payload({"prompt": "Email me at user@example.com", "nested": ["Bearer secret-token"]})

        self.assertNotIn("user@example.com", str(payload))
        self.assertNotIn("secret-token", str(payload))

    def test_llm_cost_and_counters_are_recorded(self):
        record_llm_observation(latency=1.5, prompt_tokens=100, response_tokens=40)

        self.assertEqual(calculate_llm_cost(100, 40), 0.0)
        self.assertEqual(get_metrics_snapshot()["llm_requests_total"], 1.0)
        self.assertEqual(get_metrics_snapshot()["llm_prompt_tokens_total"], 100.0)

    def test_embedding_drift_detects_matching_and_mismatched_dimensions(self):
        matching = embedding_drift([[1.0, 0.0], [0.9, 0.1]], [[1.0, 0.0], [0.8, 0.2]])
        mismatched = embedding_drift([[1.0, 0.0]], [[1.0, 0.0, 0.0]])

        self.assertEqual(matching["dimension_match"], 1.0)
        self.assertGreater(matching["centroid_cosine"], 0.99)
        self.assertEqual(mismatched["dimension_match"], 0.0)

    def test_embedding_manifest_contains_version_and_backend(self):
        manifest = embedding_model_manifest("test-model", "", "fallback", 384)

        self.assertEqual(manifest["model_name"], "test-model")
        self.assertEqual(manifest["backend"], "fallback")
        self.assertEqual(manifest["dimension"], 384)

    def test_answer_quality_rewards_grounded_answers(self):
        metrics = evaluate_answer_quality("Review task 12", tool_results="Task 12 is Review task")

        self.assertEqual(metrics["answer_present"], 1.0)
        self.assertGreater(metrics["groundedness_overlap"], 0.5)
        self.assertGreater(metrics["quality_score"], 0.6)


if __name__ == "__main__":
    unittest.main()
