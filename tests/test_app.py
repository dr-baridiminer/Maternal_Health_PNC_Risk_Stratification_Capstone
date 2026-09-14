"""Synthetic-only artifact and Streamlit smoke checks."""

import unittest
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

from app import load_artifacts, predict


class PrototypeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline, cls.metadata, cls.categories = load_artifacts()
        cls.values = {name: options[0] for name, options in cls.categories.items()}
        cls.values.update(maternal_age=28, children_ever_born=2,
                          first_anc_month=4, anc_visits=5)

    def test_saved_model_and_threshold_boundary(self):
        probability, flagged = predict(self.pipeline, self.metadata, self.values)
        self.assertTrue(0 <= probability <= 1)
        self.assertEqual(flagged, probability >= self.metadata["decision_threshold"])
        # Equality must flag; a threshold above the same score must not.
        for threshold, expected in [(probability, True), (probability + 1e-8, False)]:
            metadata = dict(self.metadata, decision_threshold=threshold)
            self.assertEqual(predict(self.pipeline, metadata, self.values)[1], expected)

    def test_missing_anc_and_all_fitted_categories(self):
        values = dict(self.values, first_anc_month=np.nan, anc_visits=np.nan)
        self.assertTrue(np.isfinite(predict(self.pipeline, self.metadata, values)[0]))
        for field, options in self.categories.items():
            for option in options:
                self.assertTrue(np.isfinite(predict(
                    self.pipeline, self.metadata, dict(self.values, **{field: option})
                )[0]))

    def test_form_validation_and_prediction(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
        app.run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.number_input) + len(app.selectbox), 13)
        app.button[0].click().run()
        self.assertTrue(app.error)
        self.assertEqual(len(app.metric), 0)
        for field, value in self.values.items():
            widgets = app.selectbox if field in self.categories else app.number_input
            widgets(key=field).set_value(value)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        probability, _ = predict(self.pipeline, self.metadata, self.values)
        self.assertEqual(app.metric[0].value, f"{probability:.2%}")
        app.number_input(key="anc_visits").set_value(0)
        app.button[0].click().run()
        self.assertTrue(app.error)
        self.assertEqual(len(app.metric), 0)
        app.number_input(key="first_anc_month").set_value(None)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertTrue(app.info)


if __name__ == "__main__":
    unittest.main()
