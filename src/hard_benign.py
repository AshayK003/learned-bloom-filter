"""Generate harder benign URLs that look more like malicious ones."""

import numpy as np


def generate_hard_benign_urls(n: int, seed: int = 42) -> list:
    """Generate benign URLs that are harder to distinguish from malicious.

    Includes IP-based URLs, URLs with ports, URLs with executable-looking paths.
    This makes the FPR test meaningful.
    """
    rng = np.random.RandomState(seed)

    tlds = ['com', 'org', 'net', 'io', 'dev', 'app', 'co', 'info', 'biz', 'us', 'ru', 'cn', 'tk', 'ml', 'ga']
    words = [
        'google', 'facebook', 'amazon', 'microsoft', 'apple', 'netflix',
        'twitter', 'github', 'stackoverflow', 'wikipedia', 'reddit', 'linkedin',
        'medium', 'dev', 'blog', 'shop', 'store', 'news', 'mail', 'cloud',
        'api', 'cdn', 'static', 'assets', 'images', 'video', 'docs', 'support',
        'update', 'download', 'install', 'setup', 'patch', 'upgrade',
    ]
    paths = [
        '', '/index.html', '/home', '/about', '/contact',
        '/download', '/update', '/install', '/setup',
        '/api/v1', '/api/v2', '/static', '/assets',
        '/file', '/upload', '/download', '/bin',
        '/cgi-bin', '/scripts', '/admin', '/login',
        '/wp-admin', '/phpmyadmin', '/config',
        '/backup', '/temp', '/tmp', '/cache',
    ]

    urls = []
    for _ in range(n):
        # 30% chance of being IP-based (like many malicious URLs)
        if rng.random() < 0.3:
            # IP-based benign (e.g., CDN, cloud service)
            ip = f"{rng.randint(1, 255)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}"
            # Sometimes with port
            if rng.random() < 0.3:
                port = rng.choice([8080, 8443, 9000, 3000, 5000, 8000])
                url = f"http://{ip}:{port}"
            else:
                url = f"http://{ip}"
            # Add path
            if rng.random() < 0.7:
                url += rng.choice(paths)
        else:
            # Domain-based
            scheme = 'https' if rng.random() > 0.3 else 'http'
            www = 'www.' if rng.random() > 0.5 else ''
            domain = rng.choice(words) + '.' + rng.choice(tlds)
            url = f'{scheme}://{www}{domain}'
            if rng.random() < 0.6:
                url += rng.choice(paths)

        urls.append(url)

    return urls


if __name__ == '__main__':
    urls = generate_hard_benign_urls(10)
    for u in urls:
        print(u)
