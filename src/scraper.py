import requests
import re
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
  "https://www.espn.com/nhl/team/schedule/_/name/bos/seasontype/2",
#   "https://www.espn.com/nhl/team/schedule/_/name/buf/seasontype/2",
]

client = ZyteAPI(api_key=API_KEY)

# TODO: Modify so that I do the 32 team scrape requests in parallel instead
def scrape_game(url):
    try:
        response = client.get({"url": url, "browserHtml": True})

        if response['statusCode'] == 200:
            soup = BeautifulSoup(response['browserHtml'], 'html.parser')
            anchor_links = soup.select('a.AnchorLink:not(.MatchInfo__Link)')

            game_ids = set()
            for link in anchor_links:
                href = link.get('href')
                if href and "gameId" in href:
                    game_ids.add(href)

            return game_ids
        else:
            print(f"Soup error, failed to fetch {url}: HTTP Status {response.status_code}")
            return set()
    except Exception as e:
        print(f"Zyte error, failed to scrape {url}: {e}")
        return set()

def modify_links(links):
    modified_links = []
    for link in links:
        match = re.search(r'/gameId/(\d+)/', link)
        if match:
            game_id = match.group(1)
            modified_links.append("https://www.espn.com/nhl/boxscore/_/gameId/" + game_id)
    return modified_links

def get_data(links):
    return 0

# 3. Get game data    
box_score_urls = []
# Gather game and player data
with open('all_boxscore_links.txt', 'r') as f:
    box_score_urls = {line.strip() for line in f}

game_data = get_data(box_score_urls)

# 1. Scrape game links
# all_game_links = set()
# for url in test_url:
#     links = scrape_game(url)
#     print(f"scraped {len(links)} games from {url} regular season")
#     all_game_links.update(links)
# with open('all_game_links.txt', 'w') as f:
#     for link in all_game_links:
#         f.write(link + '\n')

# # 2. Get to box score urls
# with open('all_game_links.txt', 'r') as f:
#     all_game_links = {line.strip() for line in f}

# # Extract "Full Box Score" URLs from acquired links
# box_score_urls = modify_links(all_game_links)

# with open('all_boxscore_links.txt', 'w') as f:
#     for link in box_score_urls:
#         f.write(link + '\n')


