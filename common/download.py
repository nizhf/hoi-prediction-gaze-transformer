#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
from tqdm import tqdm


def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value
        return None

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)


def download_file_from_dropbox(link, destination):
    session = requests.Session()
    response = session.get(link + "?dl=1", stream=True)
    save_response_content(response, destination)


def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    file_size = int(response.headers["Content-Length"])
    with open(destination, "wb") as f:
        with tqdm(total=file_size, unit="B", unit_scale=True, unit_divisor=1024) as t:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    t.update(len(chunk))
