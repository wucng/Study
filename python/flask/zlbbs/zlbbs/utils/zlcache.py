import memcache

cache = memcache.Client(['127.0.0.1:11211'],debug=True)

def set(key,value,timeout=60):
    return cache.set(key,value,timeout)

def get(key):
    return cache.get(key)

def delete(key):
    return cache.delete(key)

"""
import json
def set(key,value):
    json.dump({key:value},open("1.txt",'w'))

def get(key):
    return json.load(open("1.txt",'r'))[key]
"""