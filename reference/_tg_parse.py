#!/usr/bin/env python
"""Throwaway: render a Telegram HTML export to a clean text transcript.
Keeps sender, time, reply-to target, text, and media references."""
import re, sys, html
from html.parser import HTMLParser

PATH = sys.argv[1] if len(sys.argv) > 1 else "ChatExport_2026-06-18/messages.html"


class Msg:
    def __init__(self):
        self.id = None; self.sender = None; self.date = None
        self.reply_to = None; self.text = []; self.media = []


class TG(HTMLParser):
    def __init__(self):
        super().__init__()
        self.msgs = []
        self.cur = None
        self.last_sender = "?"
        self.mode = None          # what kind of div we are inside
        self.depth = 0
        self.capture_depth = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get("class", "")
        if tag == "div" and "message" in cls and "service" not in cls:
            self.cur = Msg()
            self.cur.id = a.get("id")
            if "joined" in cls:
                self.cur.sender = self.last_sender
            self.msgs.append(self.cur)
        if tag == "div" and self.cur is not None:
            if "from_name" in cls:
                self.mode = "from"; self.capture_depth = self.depth
            elif cls.startswith("text") or cls == "text":
                self.mode = "text"; self.capture_depth = self.depth
            elif "reply_to" in cls:
                self.mode = "reply"; self.capture_depth = self.depth
            elif "date" in cls and "title" in a:
                self.cur.date = a["title"]
        if tag == "a" and self.cur is not None:
            href = a.get("href", "")
            if self.mode == "reply" and href.startswith("#"):
                self.cur.reply_to = href[1:]
            if "photo" in a.get("class", "") or re.search(r"(photos/|files/|video)", href):
                self.cur.media.append(href)
        if tag == "br" and self.mode == "text" and self.cur:
            self.cur.text.append("\n")
        self.depth += 1

    def handle_endtag(self, tag):
        self.depth -= 1
        if self.mode and self.capture_depth is not None and self.depth <= self.capture_depth:
            self.mode = None; self.capture_depth = None

    def handle_data(self, data):
        d = data.strip("\n")
        if not d.strip():
            return
        if self.mode == "from" and self.cur:
            self.cur.sender = d.strip(); self.last_sender = d.strip()
        elif self.mode == "text" and self.cur:
            self.cur.text.append(d)
        elif self.mode == "reply" and self.cur and "In reply to" not in d:
            pass


p = TG()
p.feed(open(PATH, encoding="utf-8").read())

byid = {m.id: m for m in p.msgs if m.id}
def short(m):
    t = "".join(m.text).strip().replace("\n", " ")
    return (t[:60] + "...") if len(t) > 60 else t

for m in p.msgs:
    sender = m.sender or "?"
    date = m.date or ""
    head = f"[{date}] {sender}:"
    if m.reply_to and m.reply_to in byid:
        head += f"  (reply to {byid[m.reply_to].sender}: \"{short(byid[m.reply_to])}\")"
    print(head)
    body = "".join(m.text).strip()
    if body:
        for line in body.split("\n"):
            print("    " + line.strip())
    for med in m.media:
        print(f"    [MEDIA] {med}")
    print()
