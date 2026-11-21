"""Grounded retrieval helpers for the StayRAG sample listing catalogue."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REQUIRED_COLUMNS = (
    "id",
    "name",
    "city",
    "neighborhood",
    "monthly_rent",
    "deposit",
    "gender",
    "room_types",
    "food",
    "amenities",
    "rules",
    "rating",
    "description",
)

SYNONYMS = {
    "bangalore": "bengaluru",
    "wifi": "wi fi internet",
    "internet": "wifi",
    "food": "meals breakfast dinner vegetarian",
    "mess": "meals food breakfast dinner",
    "budget": "rent price monthly affordable",
    "cheap": "affordable budget rent",
    "girls": "women female",
    "ladies": "women female",
    "boys": "men male",
    "security": "cctv safety",
    "cleaning": "housekeeping",
    "gymnasium": "gym",
    "roommate": "double triple sharing",
    "shared": "double triple sharing",
    "private": "single room",
    "metro": "transit metro",
}


def load_listings(path: str | Path) -> pd.DataFrame:
    """Load and validate the PG catalogue CSV."""
    listings = pd.read_csv(path)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(listings.columns))
    if missing_columns:
        raise ValueError(f"PG data is missing required columns: {', '.join(missing_columns)}")
    if listings.empty:
        raise ValueError("The PG catalogue does not contain any listings.")

    listings = listings.loc[:, REQUIRED_COLUMNS].copy()
    for column in ("monthly_rent", "deposit", "rating"):
        listings[column] = pd.to_numeric(listings[column], errors="coerce")
        if listings[column].isna().any():
            raise ValueError(f"PG data contains an invalid value in '{column}'.")
    for column in REQUIRED_COLUMNS:
        if listings[column].isna().any():
            raise ValueError(f"PG data contains a blank value in '{column}'.")
    return listings


def _listing_document(listing: pd.Series) -> str:
    return " ".join(
        (
            f"{listing['name']} {listing['city']} {listing['neighborhood']}",
            f"monthly rent rupees {listing['monthly_rent']} deposit rupees {listing['deposit']}",
            f"{listing['gender']} {listing['room_types']} {listing['food']}",
            f"{listing['amenities']} {listing['rules']} {listing['description']}",
        )
    )


def _expand_query(query: str) -> str:
    expanded = query.casefold()
    for term, replacement in SYNONYMS.items():
        if re.search(rf"\b{re.escape(term)}\b", expanded):
            expanded = f"{expanded} {replacement}"
    return expanded


def _budget_limit(query: str) -> int | None:
    normalized = query.casefold().replace(",", "")
    patterns = (
        r"(?:under|below|less than|up to|upto|max(?:imum)?|within)\s*(?:₹|rs\.?\s*)?(\d{4,7})",
        r"(?:budget(?:\s+(?:of|is|around))?)\s*(?:₹|rs\.?\s*)?(\d{4,7})",
        r"(?:₹|rs\.?\s*)(\d{4,7})",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return int(match.group(1))
    if re.search(r"\bbudget\b", normalized):
        match = re.search(r"\b(\d{4,7})\b", normalized)
        if match:
            return int(match.group(1))
    return None


def _location_matches(query: str, listings: pd.DataFrame) -> pd.DataFrame | None:
    normalized_query = query.casefold().replace("bangalore", "bengaluru")
    cities = listings["city"].astype(str).unique()
    city_match = next(
        (city for city in cities if re.search(rf"\b{re.escape(city.casefold())}\b", normalized_query)),
        None,
    )
    neighborhoods = listings["neighborhood"].astype(str).unique()
    neighborhood_match = next(
        (
            neighborhood
            for neighborhood in sorted(neighborhoods, key=len, reverse=True)
            if re.search(rf"\b{re.escape(neighborhood.casefold())}\b", normalized_query)
        ),
        None,
    )
    if city_match is None and neighborhood_match is None:
        return None

    matches = listings
    if city_match is not None:
        matches = matches.loc[matches["city"] == city_match]
    if neighborhood_match is not None:
        matches = matches.loc[matches["neighborhood"] == neighborhood_match]
    return matches


def _preference_matches(query: str, listings: pd.DataFrame) -> pd.DataFrame | None:
    normalized_query = query.casefold()
    matches = listings

    gender_terms = {
        "Women": r"\b(?:women|woman|female|girls|ladies)\b",
        "Men": r"\b(?:men|man|male|boys)\b",
        "Co-living": r"\bco[\s-]?living\b",
    }
    requested_genders = [
        gender
        for gender, pattern in gender_terms.items()
        if re.search(pattern, normalized_query)
    ]
    if len(requested_genders) == 1:
        matches = matches.loc[matches["gender"] == requested_genders[0]]

    requested_rooms = [
        room
        for room in ("Single", "Double", "Triple")
        if re.search(rf"\b{room.casefold()}\b", normalized_query)
    ]
    if len(requested_rooms) == 1:
        room = requested_rooms[0]
        matches = matches.loc[
            matches["room_types"].str.split(";").apply(lambda rooms: room in rooms)
        ]

    wants_meals = re.search(
        r"\b(?:with|including|includes|included|provides|serves)\b.{0,20}"
        r"\b(?:meals?|food|mess|breakfast|dinner)\b",
        normalized_query,
    )
    if wants_meals:
        matches = matches.loc[matches["food"] != "Meals not included"]

    if matches is listings:
        return None
    return matches


class ListingRetriever:
    """TF-IDF and cosine-similarity search over validated listing records."""

    def __init__(self, listings: pd.DataFrame) -> None:
        if listings.empty:
            raise ValueError("Cannot create a retriever without PG listings.")
        self.listings = listings.reset_index(drop=True).copy()
        self.documents = self.listings.apply(_listing_document, axis=1).tolist()
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.document_matrix = self.vectorizer.fit_transform(self.documents)

    def retrieve(self, query: str, top_k: int = 5) -> pd.DataFrame:
        """Return the most similar listings, optionally constrained by a stated budget."""
        if not query.strip():
            return self.listings.iloc[0:0].assign(similarity_score=pd.Series(dtype=float))

        query_vector = self.vectorizer.transform([_expand_query(query)])
        scores = cosine_similarity(query_vector, self.document_matrix).ravel()
        results = self.listings.copy()
        results["similarity_score"] = scores

        location_matches = _location_matches(query, self.listings)
        if location_matches is not None:
            results = results.loc[results["id"].isin(location_matches["id"])]
        preference_matches = _preference_matches(query, self.listings)
        if preference_matches is not None:
            results = results.loc[results["id"].isin(preference_matches["id"])]
        budget = _budget_limit(query)
        if budget is not None:
            results = results.loc[results["monthly_rent"] <= budget]
        results = results.loc[results["similarity_score"] > 0]
        return results.sort_values("similarity_score", ascending=False).head(max(0, top_k))
