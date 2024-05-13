import re
import csv
import time
from bs4 import BeautifulSoup
from zyte_api import ZyteAPI, RequestError
from config import API_KEY

game_urls_2023_2024 = [
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

game_urls_2022_2023 = [
    "https://www.espn.com/nhl/team/schedule/_/name/bos/season/2023/seasontype/2",
    "https://www.espn.com/nhl/team/schedule/_/name/buf/season/2023/seasontype/2",
    "https://www.espn.com/nhl/team/schedule/_/name/det/season/2023/seasontype/2",
    "https://www.espn.com/nhl/team/schedule/_/name/fla/season/2023/seasontype/2",
    "https://www.espn.com/nhl/team/schedule/_/name/mtl/season/2023/seasontype/2",
    "https://www.espn.com/nhl/team/schedule/_/name/ott/season/2023/seasontype/2",
    "https://www.espn.com/nhl/team/schedule/_/name/tb/season/2023/seasontype/2",
    "https://www.espn.com/nhl/team/schedule/_/name/tor/season/2023/seasontype/2",
    "https://www.espn.com/nhl/team/schedule/_/name/ari/season/2023/seasontype/2",
    "https://www.espn.com/nhl/team/schedule/_/name/chi/season/2023/seasontype/2",

    "https://www.espn.com/nhl/team/schedule/_/name/col/season/2023/seasontype/2",
    "https://www.espn.com/nhl/team/schedule/_/name/dal/season/2023/seasontype/2",
    "https://www.espn.com/nhl/team/schedule/_/name/min/season/2023/seasontype/2",
    "https://www.espn.com/nhl/team/schedule/_/name/nsh/season/2023/seasontype/2",
    "https://www.espn.com/nhl/team/schedule/_/name/stl/season/2023/seasontype/2",
    "https://www.espn.com/nhl/team/schedule/_/name/wpg/season/2023/seasontype/2",
    "https://www.espn.com/nhl/team/schedule/_/name/car/season/2023/seasontype/2",
    "https://www.espn.com/nhl/team/schedule/_/name/cbj/season/2023/seasontype/2",
    "https://www.espn.com/nhl/team/schedule/_/name/nj/season/2023/seasontype/2",
    "https://www.espn.com/nhl/team/schedule/_/name/nyi/season/2023/seasontype/2",

    "https://www.espn.com/nhl/team/schedule/_/name/nyr/season/2023/seasontype/2",
    "https://www.espn.com/nhl/team/schedule/_/name/phi/season/2023/seasontype/2",
    "https://www.espn.com/nhl/team/schedule/_/name/pit/season/2023/seasontype/2",
    "https://www.espn.com/nhl/team/schedule/_/name/wsh/season/2023/seasontype/2",
    "https://www.espn.com/nhl/team/schedule/_/name/ana/season/2023/seasontype/2",
    "https://www.espn.com/nhl/team/schedule/_/name/cgy/season/2023/seasontype/2",
    "https://www.espn.com/nhl/team/schedule/_/name/edm/season/2023/seasontype/2",
    "https://www.espn.com/nhl/team/schedule/_/name/la/season/2023/seasontype/2",
    "https://www.espn.com/nhl/team/schedule/_/name/sj/season/2023/seasontype/2",
    "https://www.espn.com/nhl/team/schedule/_/name/sea/season/2023/seasontype/2",

    "https://www.espn.com/nhl/team/schedule/_/name/van/season/2023/seasontype/2",
    "https://www.espn.com/nhl/team/schedule/_/name/vgk/season/2023/seasontype/2",
]

client = ZyteAPI(api_key=API_KEY, n_conn=80)

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
    data = {}
    i = 0
    start_time = time.time()
    with client.session() as session:
        for result_or_exception in session.iter(links):
            if isinstance(result_or_exception, dict):
                game_id = result_or_exception["url"].split("/")[-1]
                try:
                    if result_or_exception['statusCode'] == 200:
                        soup = BeautifulSoup(result_or_exception['browserHtml'], 'html.parser')
                        rows = soup.find_all('tr', class_='Table__TR--sm')
                        data[game_id] = ""
                        for row in rows:
                            tds = row.find_all('td')
                            for item in tds:
                                data[game_id] += item.text + "\n"

                        meta_div = soup.find('div', class_='GameInfo__Meta')
                        # Find the first span element within the div
                        if meta_div:
                            first_span = meta_div.find('span')
                            if first_span:
                                # Extract text from the first span
                                data[game_id] += first_span.get_text() + "\n" 
                        end_time = time.time()
                        print(f"Got data for {i} games out of {len(links) - 1}")
                        print(f"Iteration took: {end_time - start_time} seconds")
                        i += 1
                    else:
                        print(f"Soup error, failed to fetch {result_or_exception['url']}: HTTP Status {result_or_exception['status_code']}")
                except Exception as e:
                    print(f"Zyte error, failed to scrape {result_or_exception['url']}: {e}")
    return data

def process_data(data):
    processed_data = {}
    for game in data:
        game_id = game
        lines = data[game].split('\n')
        game_data = {}
        game_data["Date"] = lines[-2]
        teams_found = 0
        scores_found = False
        away_roster_found = False
        away_roster = []
        home_roster = []
        for i in range(0, len(lines)):
            # get game results
            if teams_found < 2 and re.match(r'^[A-Z]{2,3}$', lines[i]):
                if teams_found == 0:
                    game_data['away_team'] = lines[i]
                else:
                    game_data['home_team'] = lines[i]
                    game_data['away_score'] = lines[i - 1]
                teams_found += 1
            
            if not scores_found and lines[i] == 'forwards':
                game_data['home_score'] = lines[i - 1]
                scores_found = True

            # get players       
            if lines[i] == 'forwards':
                j = i + 1
                while lines[j] != 'G':
                    if lines[j] != 'defensemen':
                        position = lines[j].rfind(' ')
                        line = lines[j][:position]
                        if away_roster_found == False:
                            away_roster.append(line)
                        else:
                            home_roster.append(line)
                    j += 1
                away_roster_found = True
            game_data['away_roster'] = away_roster
            game_data['home_roster'] = home_roster

            # get goalies
            if lines[i] == 'goalies':
                position = lines[i + 1].rfind(' ')
                goalie = lines[i + 1][:position]
                if len(home_roster) == 0:
                    game_data['away_goalie'] = goalie
                else:
                    game_data['home_goalie'] = goalie

        processed_data[game_id] = game_data
    return processed_data

# 1. Scrape game links
all_game_links = set()
for url in game_urls_2022_2023:
    links = scrape_game(url)
    print(f"scraped {len(links)} games from {url} regular season")
    all_game_links.update(links)
with open('2022-2023_game_links.txt', 'w') as f:
    for link in all_game_links:
        f.write(link + '\n')

# 2. Get to box score urls
with open('2022-2023_game_links.txt', 'r') as f:
    all_game_links = {line.strip() for line in f}
# Extract "Full Box Score" URLs from acquired links
box_score_urls = modify_links(all_game_links)

with open('2022-2023_boxscore_links.txt', 'w') as f:
    for link in box_score_urls:
        f.write(link + '\n')

# 3. Get game data and process it  
with open('2022-2023_boxscore_links.txt', 'r') as f:
    box_score_urls = {line.strip() for line in f}
adjusted_links = [{"url": url, "browserHtml": True} for url in box_score_urls]
game_data = get_data(adjusted_links)
processed_data = process_data(game_data)

# 4. Enter gathered data into a csv file
csv_file_path = '2022-2023_game_data.csv'
fieldnames = ['Game ID', 'Date', 'Away Team', 'Home Team', 'Away Roster', 'Home Roster', 'Away Score', 'Home Score', 'Away Goalie', 'Home Goalie']
with open(csv_file_path, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for game_id, data in processed_data.items():
        writer.writerow({
            'Game ID': game_id,
            'Date': data['Date'],
            'Away Team': data['away_team'],
            'Home Team': data['home_team'],
            'Away Roster': ', '.join(data['away_roster']),
            'Home Roster': ', '.join(data['home_roster']),
            'Away Score': data['away_score'],
            'Home Score': data['home_score'],
            'Away Goalie': data['away_goalie'],
            'Home Goalie': data['home_goalie']
        })

