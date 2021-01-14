import base64
from itertools import cycle

message = "C1USFwsWSwUSSUFXUFUGEA0UWlFNTkYOHx4NBwkSWxNGTltNVxcSFg0QQxMFSU1NVxcHBAcHWgVG TltNVxsPARoQSh8DAgRKXFJGAwsdRxMXCwwIHgZGQlJVCQMPAg4OGxcFRURVCQQADAMEBAFGQlJV CQUACARKXFJGBAcaCVZbTkYaGRxARRU="

key = bytes("prabhu.vanam", "utf8")

print(bytes(a ^ b for a, b in zip(base64.b64decode(message), cycle(key))))