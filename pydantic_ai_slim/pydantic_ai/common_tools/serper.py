from dataclasses import dataclass
from typing import Literal

import httpx
from pydantic import TypeAdapter
from typing_extensions import Any, TypedDict

from pydantic_ai.tools import Tool

__all__ = ('serper_search_tool',)

SERPER_API_URL = 'https://google.serper.dev/search'


class SerperSearchResult(TypedDict):
    """A Serper search result.

    See [Serper API documentation](https://serper.dev/docs) for more information.
    """

    title: str
    """The title of the search result."""
    link: str
    """The URL of the search result."""
    snippet: str
    """A short description/snippet of the search result."""
    position: int
    """The position of the result in the search results."""


serper_search_ta = TypeAdapter(list[SerperSearchResult])


@dataclass
class SerperSearchTool:
    """The Serper search tool."""

    api_key: str
    """The Serper API key."""

    async def __call__(
        self,
        query: str,
        search_type: Literal['search', 'news', 'images'] = 'search',
        num_results: int = 10,
        country: str | None = None,
        language: str | None = None,
    ) -> list[SerperSearchResult]:
        """Searches Google via Serper API for the given query and returns the results.

        Args:
            query: The search query to execute.
            search_type: The type of search to perform ('search', 'news', or 'images').
            num_results: The number of results to return (default: 10).
            country: The country code for localized results (e.g., 'us', 'cn').
            language: The language code for results (e.g., 'en', 'zh-cn').

        Returns:
            The search results.
        """
        payload: dict[str, Any] = {
            'q': query,
            'num': num_results,
        }
        if country:
            payload['gl'] = country
        if language:
            payload['hl'] = language

        # Determine the API endpoint based on search type
        url = SERPER_API_URL
        if search_type == 'news':
            url = 'https://google.serper.dev/news'
        elif search_type == 'images':
            url = 'https://google.serper.dev/images'

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json=payload,
                headers={
                    'X-API-KEY': self.api_key,
                    'Content-Type': 'application/json',
                },
            )
            response.raise_for_status()
            data = response.json()

        # Extract organic results (or news/images results based on type)
        if search_type == 'images':
            # Images have a different structure
            raw_results = data.get('images', [])
            results = [
                {
                    'title': r.get('title', ''),
                    'link': r.get('link', ''),
                    'snippet': r.get('source', ''),
                    'position': idx + 1,
                }
                for idx, r in enumerate(raw_results[:num_results])
            ]
        elif search_type == 'news':
            raw_results = data.get('news', [])
            results = [
                {
                    'title': r.get('title', ''),
                    'link': r.get('link', ''),
                    'snippet': r.get('snippet', ''),
                    'position': idx + 1,
                }
                for idx, r in enumerate(raw_results[:num_results])
            ]
        else:
            raw_results = data.get('organic', [])
            results = [
                {
                    'title': r.get('title', ''),
                    'link': r.get('link', ''),
                    'snippet': r.get('snippet', ''),
                    'position': r.get('position', idx + 1),
                }
                for idx, r in enumerate(raw_results[:num_results])
            ]

        return serper_search_ta.validate_python(results)


def serper_search_tool(api_key: str) -> Tool[Any]:
    """Creates a Serper search tool for Google search.

    Args:
        api_key: The Serper API key.

            You can get one by signing up at [https://serper.dev](https://serper.dev).
            Free tier includes 2,500 searches.

    Returns:
        A Tool instance that can be used with Pydantic AI agents.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai.common_tools.serper import serper_search_tool

        agent = Agent(
            'openai:gpt-4o',
            tools=[serper_search_tool('your-api-key')],
        )
        ```
    """
    return Tool[Any](
        SerperSearchTool(api_key=api_key).__call__,
        name='serper_search',
        description='Searches Google via Serper API for the given query and returns the results.',
    )
