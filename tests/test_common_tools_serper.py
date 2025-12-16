"""Tests for the Serper search tool."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from pytest_mock import MockerFixture

from pydantic_ai.common_tools.serper import (
    SerperSearchResult,
    SerperSearchTool,
    serper_search_tool,
)

pytestmark = pytest.mark.anyio


class TestSerperSearchTool:
    """Tests for SerperSearchTool class."""

    async def test_basic_search(self, mocker: MockerFixture):
        """Test basic search functionality."""
        mock_response_data = {
            'organic': [
                {
                    'title': 'Test Result 1',
                    'link': 'https://example.com/1',
                    'snippet': 'This is test result 1',
                    'position': 1,
                },
                {
                    'title': 'Test Result 2',
                    'link': 'https://example.com/2',
                    'snippet': 'This is test result 2',
                    'position': 2,
                },
            ]
        }

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_client.post = mocker.AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mocker.AsyncMock(return_value=None)

        mocker.patch('httpx.AsyncClient', return_value=mock_client)

        tool = SerperSearchTool(api_key='test-api-key')
        results = await tool('test query')

        assert len(results) == 2
        assert results[0]['title'] == 'Test Result 1'
        assert results[0]['link'] == 'https://example.com/1'
        assert results[1]['title'] == 'Test Result 2'

    async def test_search_with_country_and_language(self, mocker: MockerFixture):
        """Test search with country and language parameters."""
        mock_response_data: dict[str, Any] = {'organic': []}

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_client.post = mocker.AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mocker.AsyncMock(return_value=None)

        mocker.patch('httpx.AsyncClient', return_value=mock_client)

        tool = SerperSearchTool(api_key='test-api-key')
        await tool('test query', country='us', language='en')

        # Verify the call was made with correct parameters
        call_args = mock_client.post.call_args
        assert call_args is not None
        json_payload = call_args.kwargs['json']
        assert json_payload['gl'] == 'us'
        assert json_payload['hl'] == 'en'

    async def test_news_search(self, mocker: MockerFixture):
        """Test news search functionality."""
        mock_response_data = {
            'news': [
                {
                    'title': 'News Article 1',
                    'link': 'https://news.example.com/1',
                    'snippet': 'News snippet 1',
                },
            ]
        }

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_client.post = mocker.AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mocker.AsyncMock(return_value=None)

        mocker.patch('httpx.AsyncClient', return_value=mock_client)

        tool = SerperSearchTool(api_key='test-api-key')
        results = await tool('test query', search_type='news')

        assert len(results) == 1
        assert results[0]['title'] == 'News Article 1'

        # Verify the correct endpoint was called
        call_args = mock_client.post.call_args
        assert call_args is not None
        assert 'google.serper.dev/news' in call_args.args[0]

    async def test_images_search(self, mocker: MockerFixture):
        """Test images search functionality."""
        mock_response_data = {
            'images': [
                {
                    'title': 'Image 1',
                    'link': 'https://images.example.com/1',
                    'source': 'example.com',
                },
            ]
        }

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_client.post = mocker.AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mocker.AsyncMock(return_value=None)

        mocker.patch('httpx.AsyncClient', return_value=mock_client)

        tool = SerperSearchTool(api_key='test-api-key')
        results = await tool('test query', search_type='images')

        assert len(results) == 1
        assert results[0]['title'] == 'Image 1'
        assert results[0]['snippet'] == 'example.com'  # source becomes snippet

    async def test_num_results_limit(self, mocker: MockerFixture):
        """Test that num_results parameter limits results."""
        mock_response_data = {
            'organic': [
                {'title': f'Result {i}', 'link': f'https://example.com/{i}', 'snippet': f'Snippet {i}', 'position': i}
                for i in range(20)
            ]
        }

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_client.post = mocker.AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mocker.AsyncMock(return_value=None)

        mocker.patch('httpx.AsyncClient', return_value=mock_client)

        tool = SerperSearchTool(api_key='test-api-key')
        results = await tool('test query', num_results=5)

        assert len(results) == 5


class TestSerperSearchToolFactory:
    """Tests for serper_search_tool factory function."""

    def test_creates_tool_with_correct_name(self):
        """Test that the factory creates a tool with the correct name."""
        tool = serper_search_tool('test-api-key')
        assert tool.name == 'serper_search'

    def test_creates_tool_with_description(self):
        """Test that the factory creates a tool with a description."""
        tool = serper_search_tool('test-api-key')
        assert 'Serper' in tool.description
        assert 'Google' in tool.description


class TestSerperSearchResult:
    """Tests for SerperSearchResult TypedDict."""

    def test_result_structure(self):
        """Test that SerperSearchResult has the expected structure."""
        result: SerperSearchResult = {
            'title': 'Test Title',
            'link': 'https://example.com',
            'snippet': 'Test snippet',
            'position': 1,
        }
        assert result['title'] == 'Test Title'
        assert result['link'] == 'https://example.com'
        assert result['snippet'] == 'Test snippet'
        assert result['position'] == 1
