from typing import Iterable, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import requests
from requests.adapters import HTTPAdapter

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda *args, **kwargs: args[0]

BASE_URL = "api.the-odds-api.com/v4"
PROTOCOL = "https://"
TIMEOUT = 15  # seconds per request
MAX_WORKERS = 8  # concurrent requests to the API
MAX_RETRIES = 3  # retries on rate limiting (HTTP 429)

_thread_local = threading.local()


def _session() -> requests.Session:
    """One pooled session per thread, so connections (TLS handshakes) are reused across requests."""
    if not hasattr(_thread_local, "session"):
        session = requests.Session()
        session.mount(PROTOCOL, HTTPAdapter(pool_connections=1, pool_maxsize=1))
        _thread_local.session = session
    return _thread_local.session


def _get(url: str, params: dict) -> requests.Response:
    """GET with a timeout and exponential backoff when rate limited."""
    for attempt in range(MAX_RETRIES + 1):
        response = _session().get(url, params=params, timeout=TIMEOUT)
        if response.status_code != 429 or attempt == MAX_RETRIES:
            return response
        time.sleep(2 ** attempt)
    return response


class APIException(RuntimeError):
    def __str__(self):
        return f"('{self.args[0]}', '{self.args[1].json()['message']}')"


class AuthenticationException(APIException):
    pass


class RateLimitException(APIException):
    pass


def handle_faulty_response(response: requests.Response):
    if response.status_code == 401:
        raise AuthenticationException("Failed to authenticate with the API. is the API key valid?", response)
    elif response.status_code == 429:
        raise RateLimitException("Encountered API rate limit.", response)
    else:
        raise APIException("Unknown issue arose while trying to access the API.", response)


def get_sports(key: str) -> set[str]:
    url = f"{BASE_URL}/sports/"
    escaped_url = PROTOCOL + requests.utils.quote(url)
    querystring = {"apiKey": key}

    response = _get(escaped_url, querystring)
    if not response:
        handle_faulty_response(response)

    return {item["key"] for item in response.json()}


def get_data(key: str, sport: str, region: str = "eu"):
    url = f"{BASE_URL}/sports/{sport}/odds/"
    escaped_url = PROTOCOL + requests.utils.quote(url)
    querystring = {
        "apiKey": key,
        "regions": region,
        "oddsFormat": "decimal",
        "dateFormat": "unix"
    }

    response = _get(escaped_url, querystring)
    if not response:
        handle_faulty_response(response)

    return response.json()


def process_data(matches: Iterable, include_started_matches: bool = True) -> Generator[dict, None, None]:
    """Extracts all matches that are available and calculates some things about them, such as the time to start and
    the best available implied odds."""
    matches = tqdm(matches, desc="Checking all matches", leave=False, unit=" matches")
    for match in matches:
        start_time = int(match["commence_time"])
        if not include_started_matches and start_time < time.time():
            continue

        best_odd_per_outcome = {}
        for bookmaker in match["bookmakers"]:
            bookie_name = bookmaker["title"]
            for outcome in bookmaker["markets"][0]["outcomes"]:
                outcome_name = outcome["name"]
                odd = outcome["price"]
                if outcome_name not in best_odd_per_outcome.keys() or \
                    odd > best_odd_per_outcome[outcome_name][1]:
                    best_odd_per_outcome[outcome_name] = (bookie_name, odd)

        total_implied_odds = sum(1/i[1] for i in best_odd_per_outcome.values())
        match_name = f"{match['home_team']} v. {match['away_team']}"
        time_to_start = (start_time - time.time())/3600
        league = match["sport_key"]
        yield {
            "match_name": match_name,
            "match_start_time": start_time,
            "hours_to_start": time_to_start,
            "league": league,
            "best_outcome_odds": best_odd_per_outcome,
            "total_implied_odds": total_implied_odds,
        }


def fetch_all_matches(key: str, sports: Iterable[str], region: str) -> list:
    """Fetches the odds for every sport concurrently instead of one request at a time."""
    matches = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(get_data, key, sport, region): sport for sport in sports}
        progress = tqdm(as_completed(futures), total=len(futures), desc="Fetching odds", leave=False, unit=" sports")
        for future in progress:
            try:
                data = future.result()
            except AuthenticationException:
                raise
            except (APIException, requests.RequestException) as e:
                print(f"Skipping {futures[future]}: {e}")
                continue
            if isinstance(data, list):
                matches.extend(data)
    return matches


def get_arbitrage_opportunities(key: str, region: str, cutoff: float):
    sports = get_sports(key)
    data = fetch_all_matches(key, sports, region)
    results = process_data(data)
    arbitrage_opportunities = filter(lambda x: 0 < x["total_implied_odds"] < 1-cutoff, results)

    return arbitrage_opportunities