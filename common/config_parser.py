#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import yaml


def get_config(config_path):
    with Path(config_path).open("r") as config_file:
        configs = yaml.load(config_file, yaml.FullLoader)
    return configs
