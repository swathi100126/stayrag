import unittest
from pathlib import Path

from rag import ListingRetriever, load_listings

ROOT = Path(__file__).resolve().parents[1]


class ListingRetrieverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.listings = load_listings(ROOT / "data" / "pg_listings.csv")
        cls.retriever = ListingRetriever(cls.listings)

    def test_location_query_returns_location_matches(self) -> None:
        results = self.retriever.retrieve("women PG in Bangalore with meals")
        self.assertFalse(results.empty)
        self.assertTrue((results["city"] == "Bengaluru").all())
        self.assertTrue(results["gender"].eq("Women").any())

    def test_budget_query_excludes_stays_over_budget(self) -> None:
        results = self.retriever.retrieve("PG with meals under 10000")
        self.assertFalse(results.empty)
        self.assertTrue((results["monthly_rent"] <= 10000).all())

    def test_explicit_preferences_filter_out_conflicting_stays(self) -> None:
        results = self.retriever.retrieve(
            "women PG in Bangalore under 15000 with meals"
        )
        self.assertFalse(results.empty)
        self.assertTrue((results["city"] == "Bengaluru").all())
        self.assertTrue((results["gender"] == "Women").all())
        self.assertTrue((results["monthly_rent"] <= 15000).all())
        self.assertTrue((results["food"] != "Meals not included").all())

    def test_unmatched_query_does_not_invent_a_result(self) -> None:
        results = self.retriever.retrieve("petting zoo submarine")
        self.assertTrue(results.empty)

    def test_data_has_unique_listing_ids(self) -> None:
        self.assertEqual(self.listings["id"].nunique(), len(self.listings))


if __name__ == "__main__":
    unittest.main()
