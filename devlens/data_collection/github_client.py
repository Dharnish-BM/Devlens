import os
import time
import logging
import requests
from typing import Any, Dict, Optional
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class GitHubClient:
    """GitHub API client wrapping REST (v3) and GraphQL (v4) endpoints.
    
    Handles:
    - Authentication via GITHUB_PAT env variable
    - Rate limit monitoring & auto-sleep (X-RateLimit-Remaining)
    - Retries with exponential backoff on network and 5xx errors
    - Tracking total API calls made
    """

    REST_BASE_URL = "https://api.github.com"
    GRAPHQL_URL = "https://api.github.com/graphql"

    def __init__(self, token: Optional[str] = None):
        load_dotenv()
        self.token = token or os.getenv("GITHUB_PAT") or os.getenv("GITHUB_TOKEN")
        self.api_calls_count = 0
        self.rest_calls_count = 0
        self.graphql_calls_count = 0

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Devlens-Profiler/1.0",
            "Accept": "application/vnd.github.v3+json",
        })
        if self.token:
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
        else:
            logger.warning("No GITHUB_PAT found in environment. REST requests will be rate-limited (60/hr) and GraphQL will be unavailable.")

    def _handle_rate_limit(self, response: requests.Response) -> None:
        """Check rate limit headers and sleep if remaining calls hit 0."""
        remaining = response.headers.get("X-RateLimit-Remaining")
        reset_timestamp = response.headers.get("X-RateLimit-Reset")

        if remaining is not None:
            remaining_int = int(remaining)
            if remaining_int <= 1 and reset_timestamp:
                reset_time = int(reset_timestamp)
                sleep_duration = max(reset_time - int(time.time()) + 2, 1)
                logger.warning(f"Rate limit exhausted ({remaining} remaining). Sleeping for {sleep_duration}s until reset...")
                time.sleep(sleep_duration)

    def rest_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        max_retries: int = 5,
        backoff_factor: float = 1.0,
    ) -> Optional[Any]:
        """Execute a GET request against GitHub REST API v3 with retries and rate limit management."""
        url = endpoint if endpoint.startswith("http") else f"{self.REST_BASE_URL}/{endpoint.lstrip('/')}"
        
        for attempt in range(1, max_retries + 1):
            try:
                self.api_calls_count += 1
                self.rest_calls_count += 1
                response = self.session.get(url, params=params, timeout=15)
                self._handle_rate_limit(response)

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 202:
                    # 202 Accepted: GitHub is calculating stats asynchronously
                    logger.info(f"GitHub returned 202 for {url} (computing stats). Attempt {attempt}/{max_retries}...")
                    if attempt < max_retries:
                        time.sleep(1.0)
                        continue
                    return None
                elif response.status_code == 404:
                    logger.warning(f"Resource not found (404): {url}")
                    return None
                elif response.status_code == 403 and "rate limit" in response.text.lower():
                    reset_timestamp = response.headers.get("X-RateLimit-Reset")
                    if reset_timestamp:
                        sleep_duration = max(int(reset_timestamp) - int(time.time()) + 2, 1)
                        logger.warning(f"403 Rate Limit hit. Sleeping for {sleep_duration}s...")
                        time.sleep(sleep_duration)
                        continue
                elif response.status_code >= 500:
                    logger.warning(f"HTTP {response.status_code} server error on {url}. Retrying attempt {attempt}/{max_retries}...")
                else:
                    response.raise_for_status()

            except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
                logger.warning(f"Network error on attempt {attempt}/{max_retries} for {url}: {e}")

            if attempt < max_retries:
                sleep_time = backoff_factor * (2 ** (attempt - 1))
                time.sleep(sleep_time)

        logger.error(f"Failed to fetch REST endpoint after {max_retries} attempts: {url}")
        return None

    def graphql_request(
        self,
        query: str,
        variables: Optional[Dict[str, Any]] = None,
        max_retries: int = 5,
        backoff_factor: float = 1.0,
    ) -> Optional[Dict[str, Any]]:
        """Execute a GraphQL query against GitHub API v4."""
        if not self.token:
            logger.error("GraphQL API requires authentication. Please set GITHUB_PAT in .env.")
            return None

        payload = {"query": query, "variables": variables or {}}

        for attempt in range(1, max_retries + 1):
            try:
                self.api_calls_count += 1
                self.graphql_calls_count += 1
                response = self.session.post(self.GRAPHQL_URL, json=payload, timeout=20)
                self._handle_rate_limit(response)

                if response.status_code == 200:
                    res_json = response.json()
                    if "errors" in res_json:
                        logger.error(f"GraphQL returned errors: {res_json['errors']}")
                    return res_json.get("data")
                elif response.status_code >= 500:
                    logger.warning(f"GraphQL server error {response.status_code}. Retrying attempt {attempt}/{max_retries}...")
                else:
                    response.raise_for_status()

            except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
                logger.warning(f"GraphQL network error on attempt {attempt}/{max_retries}: {e}")

            if attempt < max_retries:
                sleep_time = backoff_factor * (2 ** (attempt - 1))
                time.sleep(sleep_time)

        logger.error(f"Failed to execute GraphQL query after {max_retries} attempts.")
        return None

    def get_api_calls_summary(self) -> Dict[str, int]:
        """Return breakdown of API calls made."""
        return {
            "total_calls": self.api_calls_count,
            "rest_calls": self.rest_calls_count,
            "graphql_calls": self.graphql_calls_count,
        }
