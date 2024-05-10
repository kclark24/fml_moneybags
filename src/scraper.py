import requests
from bs4 import BeautifulSoup
from zyte_api import ZyteAPI
import json
from config import API_KEY

game_urls = [
  "https://www.espn.com/nhl/team/schedule/_/name/bos/seasontype/2",
  "https://www.espn.com/nhl/team/schedule/_/name/buf/seasontype/2",
  "https://www.espn.com/nhl/team/schedule/_/name/det/seasontype/2",
  "https://www.espn.com/nhl/team/schedule/_/name/fla/seasontype/2",
  "https://www.espn.com/nhl/team/schedule/_/name/mtl/seasontype/2",
  "https://www.espn.com/nhl/team/schedule/_/name/ott/seasontype/2",
  "https://www.espn.com/nhl/team/schedule/_/name/tb/seasontype/2",
  "https://www.espn.com/nhl/team/schedule/_/name/tor/seasontype/2",
  "https://www.espn.com/nhl/team/schedule/_/name/ari/seasontype/2",
  "https://www.espn.com/nhl/team/schedule/_/name/chi/seasontype/2",

  "https://www.espn.com/nhl/team/schedule/_/name/col/seasontype/2",
  "https://www.espn.com/nhl/team/schedule/_/name/dal/seasontype/2",
  "https://www.espn.com/nhl/team/schedule/_/name/min/seasontype/2",
  "https://www.espn.com/nhl/team/schedule/_/name/nsh/seasontype/2",
  "https://www.espn.com/nhl/team/schedule/_/name/stl/seasontype/2",
  "https://www.espn.com/nhl/team/schedule/_/name/wpg/seasontype/2",
  "https://www.espn.com/nhl/team/schedule/_/name/car/seasontype/2",
  "https://www.espn.com/nhl/team/schedule/_/name/cbj/seasontype/2",
  "https://www.espn.com/nhl/team/schedule/_/name/nj/seasontype/2",
  "https://www.espn.com/nhl/team/schedule/_/name/nyi/seasontype/2",

  "https://www.espn.com/nhl/team/schedule/_/name/nyr/seasontype/2",
  "https://www.espn.com/nhl/team/schedule/_/name/phi/seasontype/2",
  "https://www.espn.com/nhl/team/schedule/_/name/pit/seasontype/2",
  "https://www.espn.com/nhl/team/schedule/_/name/wsh/seasontype/2",
  "https://www.espn.com/nhl/team/schedule/_/name/ana/seasontype/2",
  "https://www.espn.com/nhl/team/schedule/_/name/cgy/seasontype/2",
  "https://www.espn.com/nhl/team/schedule/_/name/edm/seasontype/2",
  "https://www.espn.com/nhl/team/schedule/_/name/la/seasontype/2",
  "https://www.espn.com/nhl/team/schedule/_/name/sj/seasontype/2",
  "https://www.espn.com/nhl/team/schedule/_/name/sea/seasontype/2",

  "https://www.espn.com/nhl/team/schedule/_/name/van/seasontype/2",
  "https://www.espn.com/nhl/team/schedule/_/name/vgk/seasontype/2",
]

test_url = [
  "https://www.espn.com/nhl/team/schedule/_/name/bos/seasontype/2"
]

client = ZyteAPI(api_key=API_KEY)

def scrape_game(url):
    try:
        # Send a GET request to the URL through Zyte Smart Proxy Manager
        response = client.get({"url": url, "httpResponseBody": True})
        if response.get('status') == 200:
            # Assuming the HTML content is in the 'body' key of the JSON response
            json_response = json.loads(response.text)
            html_content = json_response.get('body', '')

            # Parse the HTML content of the page
            soup = BeautifulSoup(html_content, 'html.parser')

            # Find the div with aria-label="Standings Page"
            standings_div = soup.find('div', {'aria-label': 'Standings Page'})

            # Collect all gameID links
            game_links = {link['href'] for link in standings_div.find_all('a', href=True) if "gameID" in link['href']} if standings_div else set()
            return game_links
        else:
            print(f"Failed to fetch {url}: HTTP Status {response.status_code}")
            return set()
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
        return set()

# Function to extract "Full Box Score" URLs from acquired links
def extract_box_score_urls(acquired_links):
    box_score_urls = []
    for url in acquired_links:
        try:
            # Send a GET request to the acquired link
            response = requests.get(url)
            response.raise_for_status()  # Raises an HTTPError for bad responses

            # Parse the HTML content of the page
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all links with text "Full Box Score"
            for link in soup.find_all('a', text='Full Box Score', href=True):
                box_score_urls.append(link['href'])

        except Exception as e:
            print(f"Failed to extract box score URL from {url}: {e}")

    return box_score_urls

# Main scraping logic
all_game_links = set()
for url in test_url:
    links = scrape_game(url)
    all_game_links.update(links)
    print(f"Found links for {url}: {links}")

# Extract "Full Box Score" URLs from acquired links
# box_score_urls = extract_box_score_urls(all_game_links)

# Output all found box score URLs
# print("All Full Box Score URLs found:", box_score_urls)

