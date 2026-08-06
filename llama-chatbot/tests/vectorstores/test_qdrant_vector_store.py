"""Unit tests for QdrantVectorStore.

All tests mock the ``qdrant_client`` SDK — no real Qdrant instance is required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.exceptions import (
    ConfigurationError,
    MissingCredentialsError,
    ProviderConnectionError,
    ProviderResponseError,
)
from src.vectorstores.qdrant_vector_store import QdrantVectorStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_settings() -> object:
    """Return a minimal settings-like object with required Qdrant fields."""

    class FakeSettings:
        qdrant_host: str = "localhost"
        qdrant_port: int = 6333
        qdrant_collection: str = "documents"

    return FakeSettings()


@pytest.fixture
def empty_settings() -> object:
    """Return a settings object with empty fields for validation tests."""

    class EmptySettings:
        qdrant_host: str = ""
        qdrant_port: int = 0
        qdrant_collection: str = ""

    return EmptySettings()


# ---------------------------------------------------------------------------
# 1. Provider creation
# ---------------------------------------------------------------------------
class TestProviderCreation:
    def test_creation_with_valid_settings(self, mock_settings: object) -> None:
        """Provider should initialise without error given valid settings."""
        store = QdrantVectorStore(mock_settings)
        assert store.provider_name == "qdrant"
        assert store._host == "localhost"
        assert store._port == 6333
        assert store._collection == "documents"

    def test_creation_with_custom_settings(self) -> None:
        """Custom QDRANT_HOST, QDRANT_PORT, and QDRANT_COLLECTION are accepted."""

        class CustomSettings:
            qdrant_host: str = "192.168.1.100"
            qdrant_port: int = 6334
            qdrant_collection: str = "custom_collection"

        store = QdrantVectorStore(CustomSettings())
        assert store._host == "192.168.1.100"
        assert store._port == 6334
        assert store._collection == "custom_collection"

    def test_collection_name_property(self, mock_settings: object) -> None:
        """collection_name property should return the configured collection name."""
        store = QdrantVectorStore(mock_settings)
        assert store.collection_name == "documents"


# ---------------------------------------------------------------------------
# 2. Configuration validation
# ---------------------------------------------------------------------------
class TestConfigurationValidation:
    def test_valid_configuration_passes(self, mock_settings: object) -> None:
        """Valid settings should not raise."""
        store = QdrantVectorStore(mock_settings)
        store.validate_configuration()  # should not raise

    def test_missing_host_raises(self, empty_settings: object) -> None:
        """Empty host should raise MissingCredentialsError."""
        store = QdrantVectorStore(empty_settings)
        with pytest.raises(MissingCredentialsError) as exc_info:
            store.validate_configuration()
        assert "host" in str(exc_info.value).lower() or "qdrant" in str(exc_info.value).lower()

    def test_missing_port_raises(self) -> None:
        """Port of 0 should raise MissingCredentialsError."""

        class NoPort:
            qdrant_host: str = "localhost"
            qdrant_port: int = 0
            qdrant_collection: str = "documents"

        store = QdrantVectorStore(NoPort())
        with pytest.raises(MissingCredentialsError) as exc_info:
            store.validate_configuration()
        assert "port" in exc_info.value.safe_message.lower()

    def test_missing_collection_raises(self) -> None:
        """Empty collection should raise MissingCredentialsError."""

        class NoCollection:
            qdrant_host: str = "localhost"
            qdrant_port: int = 6333
            qdrant_collection: str = ""

        store = QdrantVectorStore(NoCollection())
        with pytest.raises(MissingCredentialsError) as exc_info:
            store.validate_configuration()
        assert "collection" in exc_info.value.safe_message.lower()


# ---------------------------------------------------------------------------
# 3. Availability checks
# ---------------------------------------------------------------------------
class TestAvailability:
    def test_availability_success(self, mock_settings: object) -> None:
        """is_available should return True when Qdrant responds."""
        store = QdrantVectorStore(mock_settings)
        mock_client = MagicMock()

        with patch.object(store, "_get_client", return_value=mock_client):
            assert store.is_available() is True
            mock_client.get_collections.assert_called_once()

    def test_availability_failure(self, mock_settings: object) -> None:
        """is_available should return False on connection error without raising."""
        store = QdrantVectorStore(mock_settings)
        mock_client = MagicMock()
        mock_client.get_collections.side_effect = ConnectionError("refused")

        with patch.object(store, "_get_client", return_value=mock_client):
            assert store.is_available() is False

    def test_availability_no_exception_leak(self, mock_settings: object) -> None:
        """is_available should never raise, even on unexpected errors."""
        store = QdrantVectorStore(mock_settings)
        mock_client = MagicMock()
        mock_client.get_collections.side_effect = RuntimeError("unexpected")

        with patch.object(store, "_get_client", return_value=mock_client):
            assert store.is_available() is False


# ---------------------------------------------------------------------------
# 4. Create collection
# ---------------------------------------------------------------------------
class TestCreateCollection:
    def test_create_collection_success(self, mock_settings: object) -> None:
        """create_collection should call QdrantClient.create_collection."""
        store = QdrantVectorStore(mock_settings)
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False

        with patch.object(store, "_get_client", return_value=mock_client):
            store.create_collection(vector_size=384)

        mock_client.collection_exists.assert_called_once_with("documents")
        mock_client.create_collection.assert_called_once()

    def test_create_collection_skips_if_exists(self, mock_settings: object) -> None:
        """create_collection should skip if collection already exists."""
        store = QdrantVectorStore(mock_settings)
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True

        with patch.object(store, "_get_client", return_value=mock_client):
            store.create_collection(vector_size=384)

        mock_client.create_collection.assert_not_called()

    def test_create_collection_error(self, mock_settings: object) -> None:
        """create_collection should translate errors to ProviderResponseError."""
        store = QdrantVectorStore(mock_settings)
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        mock_client.create_collection.side_effect = RuntimeError("creation failed")

        with (
            patch.object(store, "_get_client", return_value=mock_client),
            pytest.raises(ProviderResponseError) as exc_info,
        ):
            store.create_collection(vector_size=384)

        assert "creation" in exc_info.value.safe_message.lower() or "collection" in exc_info.value.safe_message.lower()


# ---------------------------------------------------------------------------
# 5. Upsert
# ---------------------------------------------------------------------------
class TestUpsert:
    def test_upsert_success(self, mock_settings: object) -> None:
        """upsert should call QdrantClient.upsert with correct data."""
        store = QdrantVectorStore(mock_settings)
        mock_client = MagicMock()

        vectors = [[0.1, 0.2], [0.3, 0.4]]
        payloads = [{"text": "doc1"}, {"text": "doc2"}]

        with patch.object(store, "_get_client", return_value=mock_client):
            store.upsert(vectors=vectors, payloads=payloads)

        mock_client.upsert.assert_called_once()
        _, kwargs = mock_client.upsert.call_args
        assert kwargs["collection_name"] == "documents"

        # Each point should have id, vector, and payload
        points = kwargs["points"]
        assert len(points) == 2
        for point in points:
            assert hasattr(point, "id")
            assert hasattr(point, "vector")
            assert hasattr(point, "payload")

    def test_upsert_empty_vectors(self, mock_settings: object) -> None:
        """upsert with empty lists should be a no-op."""
        store = QdrantVectorStore(mock_settings)
        mock_client = MagicMock()

        with patch.object(store, "_get_client", return_value=mock_client):
            store.upsert(vectors=[], payloads=[])

        mock_client.upsert.assert_not_called()

    def test_upsert_mismatched_lengths(self, mock_settings: object) -> None:
        """upsert should raise ConfigurationError if vector/payload counts differ."""
        store = QdrantVectorStore(mock_settings)

        with pytest.raises(ConfigurationError) as exc_info:
            store.upsert(vectors=[[0.1]], payloads=[{"a": 1}, {"b": 2}])

        assert "mismatch" in str(exc_info.value).lower()

    def test_upsert_error(self, mock_settings: object) -> None:
        """upsert should translate errors to ProviderResponseError."""
        store = QdrantVectorStore(mock_settings)
        mock_client = MagicMock()
        mock_client.upsert.side_effect = RuntimeError("upsert failed")

        with (
            patch.object(store, "_get_client", return_value=mock_client),
            pytest.raises(ProviderResponseError),
        ):
            store.upsert(vectors=[[0.1, 0.2]], payloads=[{"text": "doc"}])


# ---------------------------------------------------------------------------
# 6. Search
# ---------------------------------------------------------------------------
class TestSearch:
    def test_search_success(self, mock_settings: object) -> None:
        """search should return results with id, score, and payload."""
        store = QdrantVectorStore(mock_settings)
        mock_client = MagicMock()

        # Mock a ScoredPoint-like return
        mock_point1 = MagicMock()
        mock_point1.id = "abc-123"
        mock_point1.score = 0.95
        mock_point1.payload = {"text": "result 1"}

        mock_point2 = MagicMock()
        mock_point2.id = "def-456"
        mock_point2.score = 0.87
        mock_point2.payload = {"text": "result 2"}

        mock_response = MagicMock()
        mock_response.points = [mock_point1, mock_point2]
        mock_client.query_points.return_value = mock_response

        with patch.object(store, "_get_client", return_value=mock_client):
            results = store.search(vector=[0.1, 0.2], limit=2)

        assert len(results) == 2
        assert results[0]["id"] == "abc-123"
        assert results[0]["score"] == 0.95
        assert results[0]["payload"] == {"text": "result 1"}
        assert results[1]["id"] == "def-456"
        assert results[1]["score"] == 0.87
        assert results[1]["payload"] == {"text": "result 2"}

        mock_client.query_points.assert_called_once_with(
            collection_name="documents",
            query=[0.1, 0.2],
            limit=2,
            query_filter=None,
        )

    def test_search_with_filter(self, mock_settings: object) -> None:
        """search should forward the filter_dict as a Qdrant query filter."""
        store = QdrantVectorStore(mock_settings)
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.points = []
        mock_client.query_points.return_value = mock_response

        with patch.object(store, "_get_client", return_value=mock_client):
            results = store.search(
                vector=[0.1, 0.2],
                limit=5,
                filter_dict={"document_id": "doc-1"},
            )

        assert results == []
        call_kwargs = mock_client.query_points.call_args[1]
        # The filter should be translated into a non-None Qdrant filter.
        assert call_kwargs["query_filter"] is not None

    def test_search_error(self, mock_settings: object) -> None:
        """search should translate errors to ProviderResponseError."""
        store = QdrantVectorStore(mock_settings)
        mock_client = MagicMock()
        mock_client.query_points.side_effect = RuntimeError("search failed")

        with (
            patch.object(store, "_get_client", return_value=mock_client),
            pytest.raises(ProviderResponseError),
        ):
            store.search(vector=[0.1, 0.2])


# ---------------------------------------------------------------------------
# 7. Delete
# ---------------------------------------------------------------------------
class TestDelete:
    def test_delete_success(self, mock_settings: object) -> None:
        """delete should call QdrantClient.delete with correct selector."""
        store = QdrantVectorStore(mock_settings)
        mock_client = MagicMock()

        with patch.object(store, "_get_client", return_value=mock_client):
            store.delete(document_id="doc-001")

        mock_client.delete.assert_called_once()
        _, kwargs = mock_client.delete.call_args
        assert kwargs["collection_name"] == "documents"

    def test_delete_error(self, mock_settings: object) -> None:
        """delete should translate errors to ProviderResponseError."""
        store = QdrantVectorStore(mock_settings)
        mock_client = MagicMock()
        mock_client.delete.side_effect = RuntimeError("delete failed")

        with (
            patch.object(store, "_get_client", return_value=mock_client),
            pytest.raises(ProviderResponseError),
        ):
            store.delete(document_id="doc-001")


# ---------------------------------------------------------------------------
# 8. Provider identity
# ---------------------------------------------------------------------------
class TestProviderIdentity:
    def test_provider_name_is_qdrant(self, mock_settings: object) -> None:
        """provider_name must match the registry key used for Qdrant."""
        store = QdrantVectorStore(mock_settings)
        assert store.provider_name == "qdrant"

    def test_provider_name_matches_expected_constant(self) -> None:
        """Ensure the provider_name property is stable and lowercase."""

        class CustomSettings:
            qdrant_host: str = "localhost"
            qdrant_port: int = 6333
            qdrant_collection: str = "test"

        store = QdrantVectorStore(CustomSettings())
        assert store.provider_name == "qdrant"
