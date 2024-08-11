import requests
from bs4 import BeautifulSoup
import networkx as nx
import matplotlib.pyplot as plt


def get_links(url):
    links = set()
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.startswith('http') and 'ea.com' in href:
                links.add(href)
    except Exception as e:
        print(f"Error fetching {url}: {e}")
    return links


def build_graph(urls):
    G = nx.DiGraph()
    for url in urls:
        links = get_links(url)
        for link in links:
            if link in urls:
                G.add_edge(url, link)
    return G


def calculate_pagerank(G):
    pagerank = nx.pagerank(G, alpha=0.85)
    return pagerank


def draw_graph(G, pagerank):
    pos = nx.spring_layout(G)  # Positioning of nodes
    plt.figure(figsize=(12, 12))

    # Draw the nodes and edges
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue', alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(G, pos, labels={n: f"{n.split('/')[-1][:10]}\n({pagerank[n]:.2f})" for n in G.nodes},
                            font_size=10)

    plt.title("PageRank of Selected Pages")
    plt.show()


def main():
    # List of 10 URLs to start with
    start_urls = [
        'https://www.ea.com/news/hiring-our-heroes',
        'https://www.ea.com/ea-play',
        'https://www.ea.com/games',
        'https://www.ea.com/sports',
        'https://www.ea.com/careers',
        'https://www.ea.com/news/2024-ea-sports-latest-tech-innovations',
        'https://www.ea.com/news/how-the-community-helps-shape-the-sims',
        'https://www.ea.com/legal',
        'https://www.ea.com/playtesting',
        'https://www.ea.com/store'
    ]

    # Build the graph from the URLs
    G = build_graph(start_urls)

    # Calculate PageRank
    pagerank = calculate_pagerank(G)

    # Print the PageRank results
    sorted_pagerank = dict(sorted(pagerank.items(), key=lambda item: item[1], reverse=True))
    for page, rank in sorted_pagerank.items():
        print(f"{page} PageRank: {rank:.4f}")

    # Draw the graph
    draw_graph(G, pagerank)


main()