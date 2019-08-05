#!/usr/bin/env python 
# -*- coding:utf-8 -*-

from event_further_process import extract_main
from add_key_entity import add_KE_relation
def HIN_main():
    extract_main()
    add_KE_relation()


if __name__ == "__main__":
    HIN_main()