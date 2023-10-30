import requests

l = []

def dfs(storyid, l):
    news_endpoint = "https://hacker-news.firebaseio.com/v0/item/{}.json"
    news_endpoint = news_endpoint.format(str(storyid))
    response = requests.get(news_endpoint).json()
    id = response['id']
    by = response['by']
    text = response.get('text', [])
    parent = storyid
    time = response['time']
    l.append([id, by, text, parent, time])
    for i in response.get('kids', []):
        dfs(i, l)
    return l

l = dfs(18344932, [])
for i in l:
    print(i)